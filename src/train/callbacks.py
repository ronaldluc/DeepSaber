from tensorflow import keras as K

from train.sequence import BeatmapSequence, OnEpochEnd
from utils.types import Config


def create_callbacks(train_seq: BeatmapSequence, config: Config):
    logdir = 'logdir'
    tensorboard_callback = K.callbacks.TensorBoard(logdir, histogram_freq=1)
    reduce_lr = K.callbacks.ReduceLROnPlateau(monitor='val_loss', min_delta=0.01, factor=0.1, patience=4, min_lr=0.0001)
    early_stop = K.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=11, verbose=0, mode='auto',
                                           baseline=None, restore_best_weights=True)

    callbacks = [tensorboard_callback, reduce_lr, early_stop, OnEpochEnd([train_seq])]

    return callbacks
