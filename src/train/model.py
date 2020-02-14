import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from train.sequence import BeatmapSequence
from utils.types import Config


def create_model(seq: BeatmapSequence, config: Config) -> Model:
    inputs = {}
    for col in seq.x_cols:
        if col in seq.categorical_cols:
            shape = seq.data[col].shape[1:]
            inputs[col] = layers.Input(shape=shape, name=col)
        if col in seq.regression_cols:
            shape = seq.data[col].shape[1:]
            inputs[col] = layers.Input(shape=shape, name=col)

    inputs_list = list(inputs.values())
    x = layers.concatenate(inputs=inputs_list, axis=-1)
    x = layers.LSTM(64, return_sequences=True)(x)
    # x = layers.LSTM(64)(x)

    outputs = {}
    for col in seq.y_cols:
        if col in seq.categorical_cols:
            shape = seq.data[col].shape[2]
            outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation='softmax'), name=col)(x)
        if col in seq.regression_cols:
            shape = seq.data[col].shape[2]
            outputs[col] = layers.TimeDistributed(layers.Dense(shape, activation='elu'), name=col)(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
