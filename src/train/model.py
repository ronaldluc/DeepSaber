import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from train.metric import create_metrics
from train.sequence import BeatmapSequence
from utils.types import Config


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

    model = Model(inputs=inputs, outputs=outputs)

    # optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=create_metrics(config)
    )

    return model
