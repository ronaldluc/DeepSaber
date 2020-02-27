from train.sequence import BeatmapSequence, OnEpochEnd
from utils.types import Config
from tensorflow import keras as K
import tensorflow as tf


def create_metrics(config: Config):
    return ['accuracy', ]


def mean_pred(y_true, y_pred):
    print('=' * 42)
    print('True labels')
    print(y_true)
    print('Pred labels')
    print(y_pred)
    return K.backend.mean(y_pred)
