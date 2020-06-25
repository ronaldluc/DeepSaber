from typing import List

import gensim
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.training import _minimize
from tensorflow.python.ops import embedding_ops

from train.learning_rate_schedule import FlatCosAnnealSchedule
from train.metric import create_metrics, CosineDistance
from train.sequence import BeatmapSequence
from utils.functions import y2action_word, create_word_mapping
from utils.types import Config


class AVSModel(Model):
    """
    Train/test step modification works only on TF2.2+.

    The reason to create AVS specific model instead of general model with metrics with multiple inputs
    is to avoid recomputing WordVec embeddings multiple times as their computation takes orders of magnitude
    more time than back propagation.
    """

    def __init__(self, config: Config, *args, **kwargs):
        super(AVSModel, self).__init__(*args, **kwargs)
        self.vector_metrics = {
            'avs_dist': CosineDistance('avs_dist'),
            'avs_l1': tf.keras.metrics.MeanAbsoluteError('avs_l1'),
            'avs_l2': tf.keras.metrics.MeanSquaredError('avs_l2'),
        }
        self.config = config
        self.word_model = gensim.models.KeyedVectors.load(str(self.config.dataset.action_word_model_path))
        self.word_id_dict = create_word_mapping(self.word_model)  # TODO: Rename
        self.embeddings = tf.convert_to_tensor(np.concatenate([np.zeros((2, self.word_model.vectors.shape[-1])),
                                                               self.word_model.vectors]))  # 0: MASK, 1: UNK

    @property
    def metrics(self) -> List:
        metrics: List = super(AVSModel, self).metrics
        return metrics + list(self.vector_metrics.values())

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
            # For custom training steps, users can just write:
            # trainable_variables = self.trainable_variables
            # gradients = tape.gradient(loss, trainable_variables)
            # self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        # The _minimize call does a few extra steps unnecessary in most cases,
        # such as loss scaling and gradient clipping.
        _minimize(self.distribute_strategy, tape, self.optimizer, loss,
                  self.trainable_variables)

        self.update_metrics(y_pred, y, sample_weight, train=True)

        return self.get_metrics_dict()

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self(x, training=False)
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)

        self.update_metrics(y_pred, y, sample_weight)

        return self.get_metrics_dict()

    def update_metrics(self, y_pred, y, sample_weight, train=False):
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        y_vec, y_pred_vec = None, None
        if 'word_vec' in y.keys():
            y_vec = y['word_vec']
            y_pred_vec = y_pred['word_vec']
        elif 'word_id' in y.keys() or (set(y.keys()) >= set(self.config.dataset.beat_elements) and not train):
            y_vec = self.avs_embedding(y)
            y_pred_vec = self.avs_embedding(y_pred)

        if y_vec is not None:
            for metric in self.vector_metrics.values():
                metric.update_state(y_vec, y_pred_vec)

    def get_metrics_dict(self):
        metrics = {m.name: m.result() for m in self.metrics}
        return metrics

    def word2word_vec(self, word):
        use_len = int(self.config.training.AVS_proxy_ratio * len(word)) + 1
        use_word = word[:use_len]
        try:
            word_vec = self.word_model[use_word.flatten()]
        except KeyError:  # Fallback for non-FastText based word embeddings
            word_vec = np.zeros((np.dot(*use_word.shape), self.word_model.vectors.shape[-1]), dtype=np.float32)
        return word_vec

    def avs_embedding(self, y):
        if 'word_id' in y:
            ids = tf.argmax(y['word_id'], axis=-1)
            y_vec = embedding_ops.embedding_lookup_v2(self.embeddings, ids)
        else:
            y_word = y2action_word(y)
            y_vec = tf.numpy_function(self.word2word_vec, [y_word], tf.float32)
        return y_vec


def name_generator(prefix):
    id_ = 0
    while True:
        yield f'{prefix}{id_}'
        id_ += 1


def forgiving_concatenate(inputs, axis=-1, **kwargs):
    """Functional interface to the `Concatenate` layer.
  Automatically changes to identity on `inputs` of length 1.

  Arguments:
      inputs: A list of input tensors (at least 2).
      axis: Concatenation axis.
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor, the concatenation of the inputs alongside axis `axis`.
  """
    if len(inputs) == 1:
        return inputs[0]
    return keras.layers.Concatenate(axis=axis, **kwargs)(inputs)


def create_model(seq: BeatmapSequence, stateful, config: Config) -> Model:
    batch_size = 1 if stateful else None
    names = name_generator('layer')

    inputs = {}
    per_stream = {}
    # basic_block_size = 512
    basic_block_size = 1024

    for col in seq.x_cols:
        if col in seq.categorical_cols:
            shape = None, *seq.shapes[col][2:]
            inputs[col] = layers.Input(batch_size=batch_size, shape=shape, name=col)
            per_stream[f'{col}_orig'] = inputs[col]
            # per_stream[col] = inputs[col]
            # for _ in range(3):
            #     per_stream[col] = layers.concatenate(inputs=[layers.Conv1D(filters=basic_block_size // (s - 2),
            #                                                                kernel_size=s,
            #                                                                activation='relu',
            #                                                                padding='causal',
            #                                                                kernel_initializer='lecun_normal',
            #                                                                name=names.__next__())(per_stream[col])
            #                                                  for s in [3, 5, 7]],
            #                                          axis=-1, name=names.__next__(), )
            #     per_stream[col] = layers.BatchNormalization(name=names.__next__(), )(per_stream[col])
            #     per_stream[col] = layers.SpatialDropout1D(0.2)(per_stream[col])
        if col in seq.regression_cols:
            shape = None, *seq.shapes[col][2:]
            inputs[col] = layers.Input(batch_size=batch_size, shape=shape, name=col)
            per_stream[f'{col}_orig'] = inputs[col]
            # per_stream[col] = inputs[col]

            # per_stream[f'{col}_orig'] = layers.Conv1D(filters=basic_block_size,
            #                                           kernel_size=1,
            #                                           activation='relu',
            #                                           padding='causal',
            #                                           kernel_initializer='lecun_normal',
            #                                           name=names.__next__())(per_stream[f'{col}_orig'])
            # per_stream[f'{col}_orig'] = layers.BatchNormalization(name=names.__next__(), )(per_stream[f'{col}_orig'])
            # per_stream[f'{col}_orig'] = layers.SpatialDropout1D(0.3, name=names.__next__(), )(per_stream[f'{col}_orig'])
            # for _ in range(3):
            #     per_stream[col] = layers.concatenate(inputs=[layers.Conv1D(filters=basic_block_size // (s - 2),
            #                                                                kernel_size=s,
            #                                                                activation='relu',
            #                                                                padding='causal',
            #                                                                kernel_initializer='lecun_normal',
            #                                                                name=names.__next__())(per_stream[col])
            #                                                  for s in [3, 5, 7]],
            #                                          axis=-1, name=names.__next__(), )
            #     per_stream[col] = layers.BatchNormalization(name=names.__next__(), )(per_stream[col])
            #     per_stream[col] = layers.SpatialDropout1D(0.2)(per_stream[col])

    per_stream_list = list(per_stream.values())
    x = forgiving_concatenate(inputs=per_stream_list, axis=-1, name=names.__next__(), )
    x = layers.Conv1D(filters=basic_block_size,
                      kernel_size=1,
                      activation=tfa.activations.mish,
                      padding='causal',
                      kernel_initializer='lecun_normal',
                      name=names.__next__())(x)
    x = layers.SpatialDropout1D(0.3)(x)
    # skip = x
    # for _ in range(3):
    #     x = layers.Conv1D(filters=basic_block_size,
    #                       kernel_size=1,
    #                       activation='relu',
    #                       padding='causal',
    #                       kernel_initializer='lecun_normal',
    #                       name=names.__next__())(x)
    #     x = layers.BatchNormalization(name=names.__next__(), )(x)
    #     x = layers.SpatialDropout1D(0.2)(x)
    # x = layers.concatenate(inputs=[skip, x], axis=-1, name=names.__next__(), )
    # x = layers.Dropout(0.4)(x)
    x = layers.LSTM(basic_block_size, return_sequences=True, stateful=stateful, name=names.__next__(), )(x)
    x = layers.BatchNormalization(name=names.__next__(), )(x)
    x = layers.Dropout(0.4)(x)
    x = layers.LSTM(basic_block_size, return_sequences=True, stateful=stateful, name=names.__next__(), )(x)
    x = layers.BatchNormalization(name=names.__next__(), )(x)

    outputs = {}
    loss = {}
    for col in seq.y_cols:
        if col in seq.categorical_cols:
            shape = seq.shapes[col][-1]
            outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation='softmax'), name=col)(x)
            loss[col] = keras.losses.CategoricalCrossentropy(
                label_smoothing=tf.cast(config.training.label_smoothing, 'float32'),  # TODO: Enable
            )
            # does not work with mixed precision and stateful model
        if col in seq.regression_cols:
            shape = seq.shapes[col][-1]
            outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation=None), name=col)(x)
            loss[col] = 'mse'

    if stateful or config.training.AVS_proxy_ratio == 0:
        model = Model(inputs=inputs, outputs=outputs)
        opt = keras.optimizers.Adam()
    else:
        model = AVSModel(inputs=inputs, outputs=outputs, config=config)

        # lr_schedule = tfa.optimizers.TriangularCyclicalLearningRate(
        #     initial_learning_rate=1e-4,
        #     maximal_learning_rate=8e-3,
        #     step_size=2000,
        #     scale_mode="iter",
        #     name="CyclicScheduler")
        # opt = keras.optimizers.Adam(learning_rate=lr_schedule)

        lr_schedule = FlatCosAnnealSchedule(decay_start=len(seq) * 5 + 400,  # Give extra epochs to big batch_size
                                            initial_learning_rate=8e-3,
                                            decay_steps=len(seq) * 9 + 400)
        # Ranger hyper params based on https://github.com/fastai/imagenette/blob/master/2020-01-train.md
        opt = tfa.optimizers.RectifiedAdam(learning_rate=lr_schedule,
                                           beta_1=0.95,
                                           beta_2=0.99,
                                           epsilon=1e-6)
        opt = tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=create_metrics(config),
    )

    return model


def save_model(model, model_path, train_seq, config):
    keras.mixed_precision.experimental.set_policy('float32')
    config.training.batch_size = 1
    stateful_model = create_model(train_seq, True, config)
    plain_model = keras.Model(model.inputs, model.outputs)  # drops non-serializable metrics, etc.
    stateful_model.set_weights(plain_model.get_weights())
    plain_model.save(model_path / 'model.keras')
    stateful_model.save(model_path / 'stateful_model.keras')
    return stateful_model
