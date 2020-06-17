from tensorflow import keras as K

from utils.types import Config


def create_metrics(config: Config):
    return ['accuracy', ]


def mean_pred(y_true, y_pred):
    print('=' * 42)
    print('True labels')
    print(y_true)
    print('Pred labels')
    print(y_pred)
    return K.backend.mean(y_pred)


def my_loss(*args, **kwargs):
    print(f'Loss args: {args}')
    print(f'Loss kwargs: {kwargs}')

    def real_loss(*rargs, **rkwargs):
        print(f'Loss rargs: {rargs}')
        print(f'Loss rkwargs: {rkwargs}')

        return 0.0

    return real_loss
