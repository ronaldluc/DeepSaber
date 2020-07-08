import logging
import random
from typing import List, Callable, Optional

import gensim
import kerastuner as kt
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
from train.metric import create_metrics, CosineDistance, Perplexity
from train.sequence import BeatmapSequence
from utils.functions import y2action_word, create_word_mapping, name_generator
from utils.types import Config


def get_architecture_fn(config: Config) -> Callable[..., Model]:
    architecture = {  # TODO: Change Keys to Enum
        'baseline': baseline_model,
        'ddc': ddc_model,
        'custom': create_model,
        'clstm_tune': clstm_tuning_model,
        'multi_lstm_tune': multi_lstm_tuning_model,
        'trivial_tune': trivial_tuning_model,
    }
    return architecture[config.training.model_type]


def drop_batch(y):
    dim = tf.reduce_prod(tf.shape(y)[:-1])
    flatten_y = tf.reshape(y, [dim, -1])
    return flatten_y


class AVSModel(Model):
    """
    Train/test step modification works only on TF2.2+.

    AVSModel computes action vector space related metrics from any
    of the 3 used action data input/output representation.
    If per attribute enumeration is used, the model computes the action vectors
    only on a subset of the validation data (controlled by `config.training.AVS_proxy_ratio`).

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
        self.id_metrics = {
            'id_acc': tf.keras.metrics.CategoricalAccuracy('id_acc'),
            'id_top5': tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='id_top5'),
            'perplexity': Perplexity(),
        }
        self.config = config
        if not self.config.dataset.action_word_model_path.exists():
            raise FileNotFoundError(
                f'Could not find FastText action embeddings ({self.config.dataset.action_word_model_path})'
                f'\nGenerate the action embeddings using the jupyter notebook script, '
                f'or use the normal `keras.Model` by setting `config.training.AVS_proxy_ratio`'
                f'to 0.')
        self.word_model = gensim.models.KeyedVectors.load(str(self.config.dataset.action_word_model_path))
        self.word_id_dict = create_word_mapping(self.word_model)  # TODO: Rename
        self.embeddings = tf.convert_to_tensor(np.concatenate([np.zeros((2, self.word_model.vectors.shape[-1])),
                                                               self.word_model.vectors]))  # 0: MASK, 1: UNK
        self.normed_embeddings = tf.nn.l2_normalize(self.embeddings, axis=-1)

    @property
    def metrics(self) -> List:
        metrics: List = super(AVSModel, self).metrics
        return metrics + list(self.vector_metrics.values()) + list(self.id_metrics.values())

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
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

    def call(self, inputs, training=None, mask=None):
        return super(AVSModel, self).call(inputs, training=None, mask=None)

    def get_config(self):
        super(AVSModel, self).get_config()

    def update_metrics(self, y_pred, y, sample_weight, train=False):
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        y_vec, y_pred_vec = None, None
        if 'word_vec' in y.keys():
            y_vec = y['word_vec']
            y_pred_vec = y_pred['word_vec']
            y['word_id'] = self.word_vec2word(drop_batch(y_vec))
            y_pred['word_id'] = self.word_vec2word(drop_batch(y_pred_vec))
        elif 'word_id' in y.keys() or (set(y.keys()) >= set(self.config.dataset.beat_elements) and not train):
            y_vec = self.avs_embedding(y)
            y_pred_vec = self.avs_embedding(y_pred)

        if y_vec is not None:
            for metric in self.vector_metrics.values():
                metric.update_state(y_vec, y_pred_vec)
        if 'word_id' in y.keys():
            for metric in self.id_metrics.values():
                flatten_y = drop_batch(y['word_id'])
                flatten_y_pred = drop_batch(y_pred['word_id'])

                metric.update_state(flatten_y, flatten_y_pred)
                pass

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

    def word_vec2word(self, word_vec):
        normed_array = tf.nn.l2_normalize(word_vec, axis=-1)
        transposed = tf.transpose(self.normed_embeddings, [1, 0])
        normed_array = tf.cast(normed_array, dtype=transposed.dtype)
        cosine_similarity = tf.matmul(normed_array, transposed)
        # closest_words = tf.argmax(cosine_similarity, -1)  # shape [batch_size], type int64
        closest_words = tf.nn.softmax(cosine_similarity, -1)  # shape [batch_size, embedding], type f32
        return closest_words

    def avs_embedding(self, y):
        if 'word_id' in y:
            ids = tf.argmax(y['word_id'], axis=-1)
            y_vec = embedding_ops.embedding_lookup_v2(self.embeddings, ids)
        else:
            y_word = y2action_word(y)
            y_vec = tf.numpy_function(self.word2word_vec, [y_word], tf.float32)
        return y_vec


def forgiving_concatenate(inputs, axis=-1, **kwargs):
    """
    Functional interface to the `Concatenate` layer.
    Automatically changes to identity on `inputs` of length 1.

    Arguments:
        inputs: A list of input tensors.
        axis: Concatenation axis.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor, the concatenation of the inputs alongside axis `axis`.
    """
    if len(inputs) == 1:
        return inputs[0]
    return keras.layers.Concatenate(axis=axis, **kwargs)(inputs)


def baseline_model(seq: BeatmapSequence, stateful, config: Config) -> Model:
    batch_size = 1 if stateful else None
    names = name_generator('layer')

    inputs = {}
    per_stream = {}
    basic_block_size = config.training.model_size

    for col in seq.x_cols:
        shape = None, *seq.shapes[col][2:]
        inputs[col] = layers.Input(batch_size=batch_size, shape=shape, name=col)
        per_stream[f'{col}_orig'] = inputs[col]

    per_stream_list = list(per_stream.values())
    x = forgiving_concatenate(inputs=per_stream_list, axis=-1, name=names.__next__(), )

    for i in range(config.training.lstm_repetition):
        x = layers.LSTM(basic_block_size, return_sequences=True, stateful=stateful, name=names.__next__(), )(x)

    outputs = {}
    loss = {}
    for col in seq.y_cols:
        if col in seq.categorical_cols:
            shape = seq.shapes[col][-1]
            outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation='softmax'), name=col, )(x)
            loss[col] = keras.losses.CategoricalCrossentropy()
        if col in seq.regression_cols:
            shape = seq.shapes[col][-1]
            outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation=None), name=col)(x)
            loss[col] = 'mse'

    if stateful or config.training.AVS_proxy_ratio == 0:
        if config.training.AVS_proxy_ratio == 0:
            logging.log(logging.WARNING, f'Not using AVSModel due to '
                                         f'{config.training.AVS_proxy_ratio=}.')
        model = Model(inputs=inputs, outputs=outputs)
    else:
        model = AVSModel(inputs=inputs, outputs=outputs, config=config)

    opt = keras.optimizers.Adam(lr=config.training.initial_learning_rate)
    # opt = keras.optimizers.SGD(lr=config.training.initial_learning_rate)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=create_metrics((not stateful), config),
    )

    return model


def ddc_model(seq: BeatmapSequence, stateful, config: Config) -> Model:
    batch_size = 1 if stateful else None
    names = name_generator('layer')

    inputs = {}
    per_stream = {}
    basic_block_size = config.training.model_size
    dropout = config.training.dropout

    for col in seq.x_cols:
        shape = None, *seq.shapes[col][2:]
        inputs[col] = layers.Input(batch_size=batch_size, shape=shape, name=col)
        per_stream[f'{col}_orig'] = inputs[col]

    per_stream_list = list(per_stream.values())
    x = forgiving_concatenate(inputs=per_stream_list, axis=-1, name=names.__next__(), )

    for i in range(config.training.lstm_repetition):
        x = layers.LSTM(basic_block_size, return_sequences=True, stateful=stateful, name=names.__next__(), )(x)
        x = layers.Dropout(dropout)(x)

    outputs = {}
    loss = {}
    for col in seq.y_cols:
        if col in seq.categorical_cols:
            shape = seq.shapes[col][-1]
            outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation='softmax'), name=col, )(x)
            loss[col] = keras.losses.CategoricalCrossentropy()
        if col in seq.regression_cols:
            shape = seq.shapes[col][-1]
            outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation=None), name=col)(x)
            loss[col] = 'mse'

    if stateful or config.training.AVS_proxy_ratio == 0:
        if config.training.AVS_proxy_ratio == 0:
            logging.log(logging.WARNING, f'Not using AVSModel with superior optimizer due to '
                                         f'{config.training.AVS_proxy_ratio=}.')
        model = Model(inputs=inputs, outputs=outputs)
    else:
        model = AVSModel(inputs=inputs, outputs=outputs, config=config)

    # opt = keras.optimizers.Adam(lr=config.training.initial_learning_rate, clipnorm=5.0)
    opt = keras.optimizers.SGD(lr=config.training.initial_learning_rate, clipnorm=5.0)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=create_metrics((not stateful), config),
    )

    return model


def create_model(seq: BeatmapSequence, stateful, config: Config) -> Model:
    batch_size = 1 if stateful else None
    names = name_generator('layer')

    inputs = {}
    per_stream = {}
    # basic_block_size = 512
    basic_block_size = config.training.model_size
    dropout = config.training.dropout

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
            # per_stream[f'{col}_orig'] = inputs[col]
            per_stream[col] = inputs[col]
            for _ in range(config.training.cnn_repetition):
                per_stream[col] = layers.concatenate(inputs=[layers.Conv1D(filters=basic_block_size // (s - 2),
                                                                           kernel_size=s,
                                                                           activation=tfa.activations.mish,
                                                                           padding='causal',
                                                                           kernel_initializer='lecun_normal',
                                                                           name=names.__next__())(per_stream[col])
                                                             for s in [3, 7, ]],
                                                     axis=-1, name=names.__next__(), )
                per_stream[col] = layers.BatchNormalization(name=names.__next__(), )(per_stream[col])
                per_stream[col] = layers.SpatialDropout1D(config.training.dropout)(per_stream[col])
                # layers.Add

    per_stream_list = list(per_stream.values())
    x = forgiving_concatenate(inputs=per_stream_list, axis=-1, name=names.__next__(), )

    for i in range(config.training.lstm_repetition):
        if i > 0:
            x = layers.Dropout(dropout)(x)
        x = layers.LSTM(basic_block_size, return_sequences=True, stateful=stateful, name=names.__next__(),
                        kernel_regularizer=keras.regularizers.l2(config.training.l2_regularization), )(x)
        x = layers.BatchNormalization(name=names.__next__(), )(x)

    for i in range(config.training.dense_repetition):
        if i > 0:
            x = layers.Dropout(dropout)(x)
        x = layers.TimeDistributed(
            layers.Dense(basic_block_size, activation=tfa.activations.mish, name=names.__next__(),
                         kernel_regularizer=keras.regularizers.l2(config.training.l2_regularization), ),
            name=names.__next__())(x)

    outputs = {}
    loss = {}
    for col in seq.y_cols:
        if col in seq.categorical_cols:
            shape = seq.shapes[col][-1]
            outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation='softmax'), name=col)(x)
            loss[col] = keras.losses.CategoricalCrossentropy(
                label_smoothing=tf.cast(config.training.label_smoothing, 'float32'),
            )  # does not work well with mixed precision and stateful model
            # loss[col] = Perplexity()  # TODO: Remove
        if col in seq.regression_cols:
            shape = seq.shapes[col][-1]
            outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation=None), name=col)(x)
            loss[col] = 'mse'

    if stateful or config.training.AVS_proxy_ratio == 0:
        if config.training.AVS_proxy_ratio == 0:
            logging.log(logging.WARNING, f'Not using AVSModel with superior optimizer due to '
                                         f'{config.training.AVS_proxy_ratio=}.')
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

        lr_schedule = FlatCosAnnealSchedule(decay_start=len(seq) * 15 + 400,  # Give extra epochs to big batch_size
                                            initial_learning_rate=config.training.initial_learning_rate,
                                            decay_steps=len(seq) * 21 + 400,
                                            alpha=0.01, )
        # Ranger hyper params based on https://github.com/fastai/imagenette/blob/master/2020-01-train.md
        opt = tfa.optimizers.RectifiedAdam(learning_rate=lr_schedule,
                                           beta_1=0.95,
                                           beta_2=0.99,
                                           epsilon=1e-6)
        opt = tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=create_metrics((not stateful), config),
    )

    return model


def clstm_tuning_model(seq: BeatmapSequence, stateful, config: Config) -> Model:
    def build_model(hp: kt.HyperParameters, use_avs_model: bool = False):
        batch_size = 1 if stateful else None
        layer_names = name_generator('layer')

        inputs = {}
        per_stream = {}
        cnn_activation = {'relu': keras.activations.relu,
                          'elu': keras.activations.elu,
                          'mish': tfa.activations.mish}[hp.Choice('cnn_activation', ['relu', 'elu', 'mish'])]

        for col in seq.x_cols:
            spatial_dropout = hp.Float(f'{col}_spartial_dropout', 0.0, 0.5)
            shape = None, *seq.shapes[col][2:]
            inputs[col] = layers.Input(batch_size=batch_size, shape=shape, name=col)
            per_stream[f'{col}'] = inputs[col]
            for i in range(hp.Int(f'{col}_cnn_layers_{layer_names.__next__()}', 0, 4)):
                per_stream[col] = forgiving_concatenate(inputs=[
                    layers.Conv1D(filters=hp.Int(f'{col}_{i}_{conv_i}',
                                                 32, 256, sampling='log'),
                                  kernel_size=int(s),
                                  activation=cnn_activation,
                                  padding='causal',
                                  kernel_initializer='lecun_normal',
                                  name=layer_names.__next__())(per_stream[col])
                    for conv_i, s in enumerate(hp.Choice(f'{col}_{i}_cnn_type', ['3', '35', '37', '357']))],
                    axis=-1, name=layer_names.__next__(), )
                per_stream[col] = layers.BatchNormalization(name=layer_names.__next__(), )(per_stream[col])
                per_stream[col] = layers.SpatialDropout1D(spatial_dropout)(per_stream[col])

        per_stream_list = list(per_stream.values())
        x = forgiving_concatenate(inputs=per_stream_list, axis=-1, name=layer_names.__next__(), )

        for i in range(hp.Int(f'lstm_layers', 0, 3)):
            x = layers.LSTM(hp.Int(f'{i}_lstm_layer',
                                   32, 512, sampling='log'), return_sequences=True, stateful=stateful,
                            name=layer_names.__next__(),
                            kernel_regularizer=keras.regularizers.l2(hp.Choice(f'{i}_lstm_l2',
                                                                               [1e-2, 1e-3, 1e-4, 1e-5, 0.0]), ))(x)
            x = layers.BatchNormalization(name=layer_names.__next__(), )(x)
            x = layers.Dropout(hp.Float(f'{i}_lstm_dropout', 0.0, 0.6))(x)

        spatial_dropout = hp.Float(f'out_spartial_dropout', 0.0, 0.5)
        for i in range(hp.Int(f'out_cnn_layers_{layer_names.__next__()}', 0, 2)):
            x = forgiving_concatenate(inputs=[
                layers.Conv1D(filters=hp.Int(f'out_{i}_{conv_i}',
                                             32, 256, sampling='log'),
                              kernel_size=int(s),
                              activation=cnn_activation,
                              padding='causal',
                              kernel_initializer='lecun_normal',
                              name=layer_names.__next__())(x)
                for conv_i, s in enumerate(hp.Choice(f'out_{i}_cnn_type', ['1', '3', '35', '37', '357']))],
                axis=-1, name=layer_names.__next__(), )
            x = layers.BatchNormalization(name=layer_names.__next__(), )(x)
            x = layers.SpatialDropout1D(spatial_dropout)(x)

        outputs = {}
        loss = {}
        for col in seq.y_cols:
            if col in seq.categorical_cols:
                shape = seq.shapes[col][-1]
                outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation='softmax'), name=col)(x)
                loss[col] = keras.losses.CategoricalCrossentropy(
                    label_smoothing=tf.cast(hp.Float('label_smoothing', 0.0, 0.7), 'float32'),
                )  # does not work well with mixed precision and stateful model
            if col in seq.regression_cols:
                shape = seq.shapes[col][-1]
                outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation=None), name=col)(x)
                loss[col] = 'mse'

        if stateful or config.training.AVS_proxy_ratio == 0:
            if config.training.AVS_proxy_ratio == 0:
                logging.log(logging.WARNING, f'Not using AVSModel with superior optimizer due to '
                                             f'{config.training.AVS_proxy_ratio=}.')
            model = Model(inputs=inputs, outputs=outputs)
            opt = keras.optimizers.Adam()
        else:
            if use_avs_model:
                model = AVSModel(inputs=inputs, outputs=outputs, config=config)
            else:
                model = Model(inputs=inputs, outputs=outputs)

            lr_schedule = FlatCosAnnealSchedule(decay_start=len(seq) * 15 + 400,  # Give extra epochs to big batch_size
                                                initial_learning_rate=hp.Choice('initial_learning_rate',
                                                                                [3e-2, 1e-2, 3e-3, 1e-3]),
                                                decay_steps=len(seq) * 25 + 400,
                                                alpha=0.01, )
            # Ranger hyper params based on https://github.com/fastai/imagenette/blob/master/2020-01-train.md
            opt = tfa.optimizers.RectifiedAdam(learning_rate=lr_schedule,
                                               beta_1=0.95,
                                               beta_2=0.99,
                                               epsilon=1e-6)
            opt = tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)

        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=create_metrics((not stateful), config),
        )

        return model

    return build_model


def multi_lstm_tuning_model(seq: BeatmapSequence, stateful, config: Config) -> Model:
    def build_model(hp: kt.HyperParameters, use_avs_model: bool = False):
        batch_size = 1 if stateful else None
        layer_names = name_generator('layer')

        inputs = {}
        last_layer = []

        for col in seq.x_cols:
            shape = None, *seq.shapes[col][2:]
            inputs[col] = layers.Input(batch_size=batch_size, shape=shape, name=col)
            last_layer.append(inputs[col])

        random.seed(43)
        for i in range(hp.Int(f'lstm_layers', 2, 7)):
            outs = []
            depth = hp.Int(f'depth_{i}', 4, 64, sampling='log')
            connections = min(hp.Int(f'connections_{i}', 1, 3), len(last_layer))
            dropout = hp.Float(f'dropout_{i}', 0, 0.5)
            for width_i in range(hp.Int(f'width_{i}', 1, 16)):
                t = layers.LSTM(depth, return_sequences=True,
                                name=f'lstm{i:03}_{width_i:03}_{layer_names.__next__()}',
                                stateful=stateful, )(
                    forgiving_concatenate(random.sample(last_layer, connections), name=layer_names.__next__()))
                t = layers.BatchNormalization(name=layer_names.__next__())(t)
                t = layers.Dropout(dropout, name=layer_names.__next__())(t)
                outs.append(t)
            last_layer = outs

        x = forgiving_concatenate(last_layer)
        outputs = {}
        loss = {}
        for col in seq.y_cols:
            if col in seq.categorical_cols:
                shape = seq.shapes[col][-1]
                outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation='softmax'), name=col)(x)
                loss[col] = keras.losses.CategoricalCrossentropy(
                    label_smoothing=tf.cast(hp.Float('label_smoothing', 0.0, 0.7), 'float32'),
                )  # does not work well with mixed precision and stateful model
            if col in seq.regression_cols:
                shape = seq.shapes[col][-1]
                outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation=None), name=col)(x)
                loss[col] = 'mse'

        if stateful or config.training.AVS_proxy_ratio == 0:
            if config.training.AVS_proxy_ratio == 0:
                logging.log(logging.WARNING, f'Not using AVSModel with superior optimizer due to '
                                             f'{config.training.AVS_proxy_ratio=}.')
            model = Model(inputs=inputs, outputs=outputs)
            opt = keras.optimizers.Adam()
        else:
            if use_avs_model:
                model = AVSModel(inputs=inputs, outputs=outputs, config=config)
            else:
                model = Model(inputs=inputs, outputs=outputs)

            lr_schedule = FlatCosAnnealSchedule(decay_start=len(seq) * 30,  # Give extra epochs to big batch_size
                                                initial_learning_rate=hp.Choice('initial_learning_rate',
                                                                                [3e-2, 1e-2, 8e-3, ]),
                                                decay_steps=len(seq) * 40,
                                                alpha=0.01, )
            # Ranger hyper params based on https://github.com/fastai/imagenette/blob/master/2020-01-train.md
            opt = tfa.optimizers.RectifiedAdam(learning_rate=lr_schedule,
                                               beta_1=0.95,
                                               beta_2=0.99,
                                               epsilon=1e-6)
            opt = tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)

        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=create_metrics((not stateful), config),
        )

        return model

    return build_model


def trivial_tuning_model(seq: BeatmapSequence, stateful, config: Config) -> Callable[..., Model]:
    def build_model(hp: kt.HyperParameters, use_avs_model: bool = False) -> Model:
        batch_size = 1 if stateful else None
        layer_names = name_generator('layer')
        hyper_names = name_generator('hyper')

        inputs = {}
        per_stream = {}

        for col in seq.x_cols:
            shape = None, *seq.shapes[col][2:]
            inputs[col] = layers.Input(batch_size=batch_size, shape=shape, name=col)
            per_stream[f'{col}'] = inputs[col]

        per_stream_list = list(per_stream.values())
        x = forgiving_concatenate(inputs=per_stream_list, axis=-1, name=layer_names.__next__(), )

        for i in range(hp.Int('TEST', 2, 8)):
            x = layers.LSTM(64, return_sequences=True)(x)

        outputs = {}
        loss = {}
        for col in seq.y_cols:
            if col in seq.categorical_cols:
                shape = seq.shapes[col][-1]
                outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation='softmax'), name=col)(x)
                loss[col] = keras.losses.CategoricalCrossentropy()
            if col in seq.regression_cols:
                shape = seq.shapes[col][-1]
                outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation=None), name=col)(x)
                loss[col] = 'mse'

        if config.training.AVS_proxy_ratio == 0:
            logging.log(logging.WARNING, f'Not using AVSModel with superior optimizer due to '
                                         f'{config.training.AVS_proxy_ratio=}.')
        model = Model(inputs=inputs, outputs=outputs)
        opt = keras.optimizers.Adam()

        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=['acc'],
        )

        return model

    return build_model


def save_model(model, model_path, train_seq, config, hp: Optional[kt.HyperParameters] = None):
    keras.mixed_precision.experimental.set_policy('float32')
    config.training.batch_size = 1
    stateful_model = get_architecture_fn(config)(train_seq, True, config)
    if hp is not None:
        stateful_model = stateful_model(hp, use_avs_model=True)
    plain_model = keras.Model(model.inputs, model.outputs)  # drops non-serializable metrics, etc.
    stateful_model.set_weights(plain_model.get_weights())
    plain_model.save(model_path / 'model.keras')
    stateful_model.save(model_path / 'stateful_model.keras')
    return stateful_model
