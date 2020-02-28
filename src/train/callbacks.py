from train.sequence import BeatmapSequence, OnEpochEnd
from utils.types import Config
from tensorflow import keras as K


def create_callbacks(train_seq: BeatmapSequence, config: Config):
    logdir = 'logdir'
    tensorboard_callback = K.callbacks.TensorBoard(logdir, histogram_freq=1)
    reduce_lr = K.callbacks.ReduceLROnPlateau(monitor='val_loss', min_delta=0.2, factor=0.1, patience=4, min_lr=0.001)
    early_stop = K.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02, patience=4, verbose=0, mode='auto',
                                           baseline=None, restore_best_weights=False)

    callbacks = [tensorboard_callback, reduce_lr, early_stop, OnEpochEnd([train_seq])]

    return callbacks