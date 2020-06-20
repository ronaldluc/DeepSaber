from copy import deepcopy
from functools import cached_property

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
        self.config = config

        self.init_data(config)

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size) / float(self.snippet_size)))

    def __getitem__(self, idx):
        data_dict = {}

        size = min(self.num_snippets, (idx + 1) * self.batch_size) - idx * self.batch_size  # Mixup
        new_order = np.arange(size)
        np.random.shuffle(new_order)
        ratio = np.random.beta(0.4, 0.4, (size, 1, 1)).astype('float16')  # Mixup: https://arxiv.org/pdf/1710.09412.pdf

        for col in self.x_cols | self.y_cols:
            data_dict[col] = self.data[col][idx * self.batch_size:(idx + 1) * self.batch_size]

            if col in self.categorical_cols:    # to categorical
                num_classes = [num for ending, num in self.config.dataset.num_classes.items() if col.endswith(ending)][0]
                data_dict[col] = keras.utils.to_categorical(data_dict[col], num_classes, dtype='float16')

        for col in self.x_cols | self.y_cols:
            data_dict[col] = ratio * data_dict[col] + (1 - ratio) * data_dict[col][new_order]

        return {col: data_dict[col] for col in self.x_cols}, {col: data_dict[col] for col in self.y_cols}

    def on_epoch_end(self):
        """Mirror horizontally"""
        new_order = np.arange(self.num_snippets)
        np.random.shuffle(new_order)
        # ratio = np.random.beta(0.4, 0.4, (self.num_snippets, 1, 1))     # Mixup: https://arxiv.org/pdf/1710.09412.pdf
        for col in self.data:
            # self.data[col] = ratio * self.original_data[col] + (1 - ratio) * self.original_data[col][new_order]
            self.data[col] = self.data[col][new_order]

        pass

    @cached_property
    def shapes(self):
        x, y = self[0]
        shapes = {col: data.shape for col, data in x.items()}
        shapes.update({col: data.shape for col, data in y.items()})

        return shapes


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
                                   .tolist(), dtype='float16')
                     for col in self.x_cols | self.y_cols}

        for col in self.data:
            if len(self.data[col].shape) < 3:
                self.data[col] = self.data[col].reshape(*shape, 1)

        # self.original_data = deepcopy(self.data)

        # for col in self.categorical_cols & (self.x_cols | self.y_cols):
        #     print(f'Processing {col=}')
        #     num_classes = [num for ending, num in config.dataset.num_classes.items() if col.endswith(ending)][0]
        #     self.data[col] = keras.utils.to_categorical(self.data[col], num_classes, dtype='float32')
