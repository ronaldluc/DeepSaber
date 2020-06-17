from typing import Dict

import gensim
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter

from train.metric import create_metrics
from train.sequence import BeatmapSequence
from utils.types import Config


def compute_acc(res_dict):
    acc = [val for key, val in res_dict.items() if 'acc' in key]
    if len(acc) == 0:
        return 0.0
    return sum(acc) / len(acc)


def y2action_word(y: Dict):
    """
    Converts dictionary of action one-hot vectors into a action word representation
    Example output element: L000_R001
    """
    word = []

    for hand in 'lr':
        word += [hand.upper()]
        word += [tf.strings.as_string(tf.argmax(y[f'{hand}_{name}'], axis=-1)) for name in
                 ['lineLayer', 'lineIndex', 'cutDirection']]
        word += ['_']

    return tf.strings.join(word[:-1])


class AVSModel(Model):
    """
    Train/test step modification works only on TF2.2+.

    The reason to create AVS specific model instead of general model with metrics with multiple inputs
    is to avoid recomputing WordVec embeddings multiple times as their computation takes orders of magnitude
    more time than back propagation.
    """
    def __init__(self, config: Config, *args, **kwargs):
        super(AVSModel, self).__init__(*args, **kwargs)
        self.avs_metric = tf.keras.metrics.CosineSimilarity('avs')
        self.config = config
        self.word_model = gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.load(
            str(self.config.dataset.storage_folder / 'fasttext.model'))

    def reset_metrics(self):
        super(AVSModel, self).reset_metrics()
        self.avs_metric.reset_states()

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
            # For custom training steps, users can just write:
            trainable_variables = self.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        # The _minimize call does a few extra steps unnecessary in most cases,
        # such as loss scaling and gradient clipping.
        # _minimize(self.distribute_strategy, tape, self.optimizer, loss,
        #           self.trainable_variables)

        self.update_metrics(y_pred, y, sample_weight)

        return self.get_metrics_dict()

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self(x, training=False)
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)

        self.update_metrics(y_pred, y, sample_weight)

        return self.get_metrics_dict()

    def update_metrics(self, y_pred, y, sample_weight):
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        y_vec = self.avs_embedding(y)
        y_pred_vec = self.avs_embedding(y_pred)
        self.avs_metric.update_state(y_vec, y_pred_vec)

    def get_metrics_dict(self):
        metrics = {m.name: m.result() for m in self.metrics}
        return {'loss': metrics['loss'], 'avg_acc': compute_acc(metrics), 'avs': self.avs_metric.result()}

    def word2word_vec(self, word):
        use_len = int(self.config.training.AVS_proxy_ratio * len(word)) + 1
        use_word = word[:use_len]
        word_vec = self.word_model[use_word.flatten()]
        new_shape = *use_word.shape, word_vec.shape[-1]
        return np.reshape(word_vec, new_shape)

    def avs_embedding(self, y):
        y_word = y2action_word(y)
        y_vec = tf.numpy_function(self.word2word_vec, [y_word], tf.float32)
        return y_vec


def create_model(seq: BeatmapSequence, stateful, config: Config) -> Model:
    batch_size = 1 if stateful else None

    inputs = {}
    per_stream = {}
    for col in seq.x_cols:
        if col in seq.categorical_cols:
            shape = None, *seq.data[col].shape[2:]
            inputs[col] = layers.Input(batch_size=batch_size, shape=shape, name=col)
            per_stream[col] = layers.concatenate(inputs=[layers.Conv1D(filters=64 // (s - 2),
                                                                       kernel_size=s,
                                                                       activation='relu',
                                                                       padding='causal')(inputs[col])
                                                         for s in [3, 5, 7]],
                                                 axis=-1)
            per_stream[col] = inputs[col]
            per_stream[col] = layers.BatchNormalization()(per_stream[col])
        if col in seq.regression_cols:
            shape = None, *seq.data[col].shape[2:]
            inputs[col] = layers.Input(batch_size=batch_size, shape=shape, name=col)
            per_stream[col] = layers.concatenate(inputs=[layers.Conv1D(filters=128 // (s - 2),
                                                                       kernel_size=s,
                                                                       activation='relu',
                                                                       padding='causal')(inputs[col])
                                                         for s in [3, 5, 7]],
                                                 axis=-1)
            per_stream[col] = layers.BatchNormalization()(per_stream[col])

    per_stream_list = list(per_stream.values())
    x = layers.concatenate(inputs=per_stream_list, axis=-1)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(256, return_sequences=True, stateful=stateful)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(128, return_sequences=True, stateful=stateful)(x)

    outputs = {}
    loss = {}
    for col in seq.y_cols:
        if col in seq.categorical_cols:
            shape = seq.data[col].shape[-1]
            outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation='softmax'), name=col)(x)
            loss[col] = keras.losses.CategoricalCrossentropy(
                label_smoothing=tf.cast(config.training.label_smoothing, 'float32'))
            # does not work with mixed precision and stateful model
        if col in seq.regression_cols:
            shape = seq.data[col].shape[-1]
            outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation='elu'), name=col)(x)
            loss[col] = 'mae'

    if stateful:
        model = Model(inputs=inputs, outputs=outputs)
    else:
        model = AVSModel(inputs=inputs, outputs=outputs, config=config)

    # optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        # loss=my_loss,
        metrics=create_metrics(config)  # + [my_loss(outputs)]
    )

    return model
