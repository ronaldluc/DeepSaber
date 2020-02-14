import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow import keras

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
        df = df.reset_index('difficulty')
        df = df[df['difficulty'].isin(config.training['use_difficulties'])]
        df['difficulty'] = df['difficulty'].replace(config.dataset['difficulty_mapping'])

        self.df = df
        self.batch_size = config.training['batch_size']
        self.snippet_size = config.beat_preprocessing['snippet_window_length']

        self.data = {col: np.array(df[col]
                                   .to_numpy()
                                   .reshape((len(df) // self.snippet_size, self.snippet_size))
                                   .tolist())
                     for col in df.columns}
        self.categorical_cols = sum([config.dataset[name] for name in config.training['categorical']], [])
        self.regression_cols = sum([config.dataset[name] for name in config.training['regression']], [])

        for col in self.categorical_cols:
            num_classes = [num for ending, num in config.dataset['num_classes'].items() if col.endswith(ending)][0]
            self.data[col] = keras.utils.to_categorical(self.data[col], num_classes, dtype='float32')
        pass

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size) / float(self.snippet_size)))

    def __getitem__(self, idx):
        x = {}
        for col in
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        start = 0

        return None
