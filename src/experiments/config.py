import gc
import multiprocessing
import os
from typing import Dict

os.environ['AUTOGRAPH_VERBOSITY'] = '5'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random

import numpy as np

import tensorflow as tf

from process.api import create_song_list, load_datasets
from train.callbacks import create_callbacks
from train.model import create_model
from train.sequence import BeatmapSequence
from utils.types import Config, Timer, DatasetConfig
import pandas as pd


def main():
    timer = Timer()

    seed = 43  # random, non-fine tuned seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    config = Config()

    base_folder = config.base_data_folder
    song_folders = create_song_list(config.dataset.beat_maps_folder)
    total = len(song_folders)
    print(f'Found {total} folders')

    config.dataset.storage_folder = base_folder / 'new_datasets'
    # config.dataset.storage_folder = base_folder / 'test_datasets'
    # config.audio_processing.use_cache = False
    # generate_datasets(song_folders, config)

    train, val, test = load_datasets(config)
    timer('Loaded datasets', 0)

    # Ensure this song is excluded from the training data for hand tasting
    train.drop(index='133b', inplace=True, errors='ignore')
    train.drop(index='Daddy - PSY', inplace=True, errors='ignore')
    # dataset_stats(train)

    manager = multiprocessing.Manager()
    return_list = manager.list()

    for repetition in range(7):
        # hyper_params = {'mixup_alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25]}
        # config = Config()
        # config.training.label_smoothing = 0
        # eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config)
        #
        # hyper_params = {'label_smoothing': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
        # config = Config()
        # config.training.mixup_alpha = 0
        # eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config)
        #
        # hyper_params = {'label_smoothing': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75,],
        #                 'mixup_alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75,]}
        # config = Config()
        # eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config)

        # hyper_params = {'batch_size': [1024, 512, 256, 128, 64, 32]}
        # config = Config()
        # config.training.mixup_alpha = 0.75
        # config.training.label_smoothing = 0
        # eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config)

        hyper_params = {'x_groups': [
            # Without previous beat
            [DatasetConfig.categorical, DatasetConfig.audio, DatasetConfig.regression],
            # Without ActionVec information
            [['prev_word_id'], DatasetConfig.categorical, DatasetConfig.audio, DatasetConfig.regression],
            [['prev_word_id'], DatasetConfig.categorical, DatasetConfig.audio, ],
            [['prev_word_id'], DatasetConfig.categorical, DatasetConfig.regression],
            [['prev_word_id'], DatasetConfig.audio, DatasetConfig.regression],
            [['prev_word_id'], ],
            # Without one data stream
            [['prev_word_vec'], ],
            [['prev_word_vec'], DatasetConfig.categorical, DatasetConfig.audio, DatasetConfig.regression],
            [['prev_word_vec'], DatasetConfig.categorical, DatasetConfig.audio, ],
            [['prev_word_vec'], DatasetConfig.categorical, DatasetConfig.regression],
            [['prev_word_vec'], DatasetConfig.audio, DatasetConfig.regression],
        ]}
        config = Config()
        config.training.mixup_alpha = 0.5
        config.training.label_smoothing = 0.5
        eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config)


def eval_hyperparams(base_folder, timer, hyper_params: Dict, return_list, train, val, test, config):
    test_name = ':'.join(hyper_params.keys())
    for parameters in zip(*hyper_params.values()):
        print(f'{test_name} = {parameters} | ' * 10)
        for hyper_param, parameter in zip(hyper_params, parameters):
            setattr(config.training, hyper_param, parameter)

        return_list[:] = []
        process = multiprocessing.Process(target=get_config_model_loss,
                                          args=(train, val, test, config, return_list))
        process.start()
        process.join()
        history, eval_metrics = return_list
        eval_metrics['history'] = history
        eval_metrics['elapsed'] = timer(f'{parameters} evaluated')
        series = pd.Series(eval_metrics, name=':'.join([str(x) for x in parameters]))

        if (base_folder / 'temp' / f'{test_name}.csv').exists():
            df = pd.read_csv(base_folder / 'temp' / f'{test_name}.csv', index_col=0)
            df = df.append(series)
        else:
            df = pd.DataFrame([series, ])
        df.index.name = test_name
        print(df)
        df.to_csv(base_folder / 'temp' / f'{test_name}.csv')


def get_config_model_loss(train, val, test, config, return_list):
    train_seq = BeatmapSequence(df=train, is_train=True, config=config)
    val_seq = BeatmapSequence(df=val, is_train=False, config=config)
    test_seq = BeatmapSequence(df=test, is_train=False, config=config)

    # keras.mixed_precision.experimental.set_policy('mixed_float16')    # breaks loss for more advanced models
    model = create_model(train_seq, False, config)

    callbacks = create_callbacks(train_seq, config)
    # callbacks = []

    history = model.fit(train_seq,
                        validation_data=val_seq,
                        callbacks=callbacks,
                        epochs=150,
                        verbose=0,
                        workers=10,
                        max_queue_size=16,
                        use_multiprocessing=False,
                        )
    eval_metrics = model.evaluate(test_seq, workers=10, return_dict=True, verbose=0)

    tf.keras.backend.clear_session()
    del train_seq, val_seq, test_seq
    gc.collect()

    print(f'{return_list=}')
    return_list[:] = (history.history, eval_metrics)
    print(f'{return_list=}')


if __name__ == '__main__':
    main()
