from tensorflow import keras as K

from train.sequence import BeatmapSequence, OnEpochEnd
from utils.types import Config


def create_callbacks(train_seq: BeatmapSequence, config: Config):
    logdir = 'logdir'
    callbacks = [
        K.callbacks.TensorBoard(logdir, histogram_freq=1),
        # K.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=0.001, factor=0.7, patience=10, min_lr=0.00005,
        #                               verbose=1, cooldown=3),
        K.callbacks.EarlyStopping(monitor='avg_acc', min_delta=0.000, patience=5, verbose=1, mode='auto',
                                  baseline=None, restore_best_weights=True),
        OnEpochEnd([train_seq]),
    ]

    return callbacks
