import gc
import multiprocessing
import os

os.environ['AUTOGRAPH_VERBOSITY'] = '5'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random

import numpy as np

import tensorflow as tf

from process.api import create_song_list, load_datasets
from train.callbacks import create_callbacks
from train.model import create_model
from train.sequence import BeatmapSequence
from utils.types import Config, Timer
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

    for repetition in range(43):
        for hyper_param, param_range in [('mixup_alpha', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]),
                                         ('label_smoothing', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                                         ('batch_size', [1024, 512, 256, 128, 64, 32])]:
            for parameter in param_range:
                print(f'{hyper_param} = {parameter:5} | ' * 10)
                setattr(config.training, hyper_param, parameter)
                return_list[:] = []
                process = multiprocessing.Process(target=get_config_model_loss,
                                                  args=(train, val, test, config, return_list))
                process.start()
                process.join()
                history, eval_metrics = return_list
                eval_metrics['history'] = history
                eval_metrics['elapsed'] = timer(f'{parameter} evaluated')

                df = pd.read_csv(base_folder / 'temp' / f'{hyper_param}.csv', index_col=0)
                df = df.append(pd.Series(eval_metrics, name=parameter))
                df.index.name = hyper_param
                print(df)
                df.to_csv(base_folder / 'temp' / f'{hyper_param}.csv')

    print(df)


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
