import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from train.metric import create_metrics
from train.sequence import BeatmapSequence
from utils.types import Config


def create_model(seq: BeatmapSequence, stateful, config: Config) -> Model:
    batch_size = 1 if stateful else None

    inputs = {}
    for col in seq.x_cols:
        if col in seq.categorical_cols:
            shape = None, *seq.data[col].shape[2:]
            inputs[col] = layers.Input(batch_size=batch_size, shape=shape, name=col)
        if col in seq.regression_cols:
            shape = None, *seq.data[col].shape[2:]
            inputs[col] = layers.Input(batch_size=batch_size, shape=shape, name=col)

    inputs_list = list(inputs.values())
    x = layers.concatenate(inputs=inputs_list, axis=-1)
    # x = layers.LSTM(1024, return_sequences=True)(x)
    x = layers.LSTM(32, return_sequences=True, stateful=stateful)(x)
    # x = layers.LSTM(64, return_sequences=True)(x)
    # x = layers.LSTM(256, return_sequences=True)(x)
    # x = layers.LSTM(64)(x)

    outputs = {}
    for col in seq.y_cols:
        if col in seq.categorical_cols:
            shape = seq.data[col].shape[-1]
            outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation='softmax'), name=col)(x)
        if col in seq.regression_cols:
            shape = seq.data[col].shape[-1]
            outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation='elu'), name=col)(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=create_metrics(config)
    )

    return model
