import logging
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

    def __init__(self, df: pd.DataFrame, is_train: bool, config: Config):
        df = add_difficulty(df, config)

        self.df_len = len(df)
        self.batch_size = config.training.batch_size
        self.snippet_size = config.beat_preprocessing.snippet_window_length
        self.config = config
        self.is_train = is_train

        self.init_data(df, config)

    def __len__(self):
        return int(np.ceil(self.df_len / float(self.batch_size) / float(self.snippet_size)))

    def __getitem__(self, idx):
        data_dict = {}

        for col in self.x_cols | self.y_cols:
            data_dict[col] = self.data[col][idx * self.batch_size:(idx + 1) * self.batch_size]

            if col in self.categorical_cols:  # to categorical
                num_classes = [num for ending, num in self.config.dataset.num_classes.items() if col.endswith(ending)][
                    0]
                data_dict[col] = keras.utils.to_categorical(data_dict[col], num_classes, dtype='float32')

        if self.is_train and self.config.training.mixup_alpha >= 1e-4:  # Mixup: https://arxiv.org/pdf/1710.09412.pdf
            size = min(self.num_snippets, (idx + 1) * self.batch_size) - idx * self.batch_size
            new_order = np.arange(size)
            np.random.shuffle(new_order)
            ratio = np.random.beta(self.config.training.mixup_alpha, self.config.training.mixup_alpha,
                                   (size, 1, 1)).astype('float32')

            for col in self.x_cols | self.y_cols:
                data_dict[col] = ratio * data_dict[col] + (1 - ratio) * data_dict[col][new_order]

        return {col: data_dict[col] for col in self.x_cols}, {col: data_dict[col] for col in self.y_cols}

    def on_epoch_end(self):
        """Shuffle the data to make new Mixups possible"""
        new_order = np.arange(self.num_snippets)
        np.random.shuffle(new_order)
        for col in self.data:
            self.data[col] = self.data[col][new_order]

    @cached_property
    def shapes(self):
        x, y = self[0]
        shapes = {col: data.shape for col, data in x.items()}
        shapes.update({col: data.shape for col, data in y.items()})

        return shapes

    def init_data(self, df, config: Config):
        """Makes Sequence data representation re-inializable with a different Config"""
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
                                   .tolist(), dtype='float32')
                     for col in self.categorical_cols | self.regression_cols}

        for col in self.data:
            if len(self.data[col].shape) < 3:
                self.data[col] = self.data[col].reshape(*shape, 1)

        if self.data['word_id'].max() == 0 and 'word_id' in ' '.join(self.shapes.keys()):
            logging.log(logging.ERROR, f'Using action vector space information without loaded FastText action '
                                       f'embeddings. The embeddings should be in '
                                       f'{config.dataset.action_word_model_path}')
