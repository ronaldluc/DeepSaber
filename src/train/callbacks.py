from datetime import datetime

import numpy as np
from tensorflow import keras as K

from train.sequence import BeatmapSequence, OnEpochEnd
from utils.types import Config


def create_callbacks(train_seq: BeatmapSequence, config: Config):
    logdir = f'../data/logdir1/model_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    callbacks = [
        # K.callbacks.TensorBoard(logdir, histogram_freq=0),    # Slows auto search. Enable if experimenting by hand.
        # ForgivingEarlyStopping(monitor='val_avs_dist', max_forgiveness=0.003, patience=8, verbose=0, mode='auto',
        #                        baseline=None, restore_best_weights=True),
        K.callbacks.EarlyStopping(monitor='val_avs_dist', min_delta=0.001, patience=7, verbose=0, mode='auto',
                                  baseline=None, restore_best_weights=True),
        OnEpochEnd([train_seq]),
    ]

    return callbacks


class ForgivingEarlyStopping(K.callbacks.EarlyStopping):
    """Stop training when a monitored metric has worsen significantly (over `max_delta`).

    Arguments:
    monitor: Quantity to be monitored.
    min_delta: Minimum change in the monitored quantity
        to qualify as an improvement, i.e. an absolute
        change of less than min_delta, will count as no
        improvement.
    max_forgiveness: Maximum negative change in the monitored quantity
        to qualify as an deterioration, i.e. an absolute
        change of more than max_forgiveness, will count as deterioration.
    patience: Number of epochs with deterioration
        after which training will be stopped.
    verbose: verbosity mode.
    mode: One of `{"auto", "min", "max"}`. In `min` mode,
        training will stop when the quantity
        monitored has stopped decreasing; in `"max"`
        mode it will stop when the quantity
        monitored has stopped increasing; in `"auto"`
        mode, the direction is automatically inferred
        from the name of the monitored quantity.
    baseline: Baseline value for the monitored quantity.
        Training will stop if the model doesn't show improvement over the
        baseline.
    restore_best_weights: Whether to restore model weights from
        the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of
        training are used.

    """

    def __init__(self,
                 monitor='val_loss',
                 max_forgiveness=None,
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(ForgivingEarlyStopping, self).__init__(monitor=monitor,
                                                     patience=patience,
                                                     verbose=verbose,
                                                     min_delta=min_delta,
                                                     mode=mode,
                                                     baseline=baseline,
                                                     restore_best_weights=restore_best_weights)
        self.max_forgiveness = max_forgiveness
        if self.max_forgiveness:
            self.max_forgiveness = -abs(self.max_forgiveness)
            if self.monitor_op == np.greater:
                self.max_forgiveness *= 1
            else:
                self.max_forgiveness *= -1

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        elif self.max_forgiveness and self.monitor_op(current - self.max_forgiveness, self.best):
            self.wait += 0.0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)
