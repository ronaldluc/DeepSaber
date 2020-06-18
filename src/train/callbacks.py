from datetime import datetime

from tensorflow import keras as K

from train.sequence import BeatmapSequence, OnEpochEnd
from utils.types import Config


def create_callbacks(train_seq: BeatmapSequence, config: Config):
    logdir = f'../data/logdir1/model_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    callbacks = [
        K.callbacks.TensorBoard(logdir, histogram_freq=5),
        # K.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=0.001, factor=0.7, patience=10, min_lr=0.00005,
        #                               verbose=1, cooldown=3),
        K.callbacks.EarlyStopping(monitor='val_avs_dist', min_delta=0.0, patience=10, verbose=0, mode='auto',
                                  baseline=None, restore_best_weights=True),
        OnEpochEnd([train_seq]),
    ]

    return callbacks
