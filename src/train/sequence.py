from copy import deepcopy

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.utils import Sequence

from train.compute import add_difficulty
from utils.types import Config


class OnEpochEnd(keras.callbacks.Callback):
    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end()


class BeatmapSequence(Sequence):

    def __init__(self, df: pd.DataFrame, config: Config):
        df = add_difficulty(df, config)

        self.df = df
        self.batch_size = config.training.batch_size
        self.snippet_size = config.beat_preprocessing.snippet_window_length

        self.init_data(config)

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size) / float(self.snippet_size)))

    def __getitem__(self, idx):
        x = {}
        for col in self.x_cols:
            x[col] = self.data[col][idx * self.batch_size:(idx + 1) * self.batch_size]
        y = {}
        for col in self.y_cols:
            y[col] = self.data[col][idx * self.batch_size:(idx + 1) * self.batch_size]

        return x, y

    def on_epoch_end(self):
        """Mirror horizontally"""
        new_order = np.arange(self.num_snippets)
        np.random.shuffle(new_order)
        ratio = np.random.beta(0.4, 0.4, (self.num_snippets, 1, 1))     # Mixup: https://arxiv.org/pdf/1710.09412.pdf
        for col in self.data:
            self.data[col] = ratio * self.original_data[col] + (1 - ratio) * self.original_data[col][new_order]
            self.data[col] = self.data[col][new_order]

        pass

    def init_data(self, config: Config):
        """Makes Sequence data representation re-inializable with a different Config"""
        df = self.df
        self.num_snippets = max(1, len(df) // self.snippet_size)
        shape = self.num_snippets, min(len(df), self.snippet_size)
        # shape == (number of snippets, snippet size)

        self.categorical_cols = set(sum([list(cols) for cols in config.training.categorical_groups], []))
        self.regression_cols = set(sum([list(cols) for cols in config.training.regression_groups], []))
        self.x_cols = set(sum([list(cols) for cols in config.training.x_groups], []))
        self.y_cols = set(sum([list(cols) for cols in config.training.y_groups], []))

        self.data = {col: np.array(df[col]
                                   .to_numpy()
                                   .reshape(shape)
                                   .tolist(), dtype=np.float_)
                     for col in self.x_cols | self.y_cols}

        for col in self.data:
            if len(self.data[col].shape) < 3:
                self.data[col] = self.data[col].reshape(*shape, 1)

        for col in self.categorical_cols:
            num_classes = [num for ending, num in config.dataset.num_classes.items() if col.endswith(ending)][0]
            self.data[col] = keras.utils.to_categorical(self.data[col], num_classes, dtype=np.float_)

        self.original_data = deepcopy(self.data)
