import gc
import multiprocessing
import random
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from process.api import create_song_list, load_datasets
from train.callbacks import create_callbacks
from train.model import get_architecture_fn
from train.sequence import BeatmapSequence
from utils.types import Timer, Config


def eval_hyperparams(base_folder, timer, hyper_params: Dict, return_list, train, val, test, config, prefix=''):
    test_name = ':'.join(hyper_params.keys())
    csv_file = base_folder / 'temp' / f'{prefix}{test_name}.csv'

    for parameters in zip(*hyper_params.values()):
        print(f'{test_name} = {parameters} | ' * 4)
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

        if csv_file.exists():
            df = pd.read_csv(csv_file, index_col=0)
            df = df.append(series)
        else:
            df = pd.DataFrame([series, ])
        df.index.name = test_name
        df.to_csv(csv_file)


def get_config_model_loss(train, val, test, config, return_list):
    train_seq = BeatmapSequence(df=train, is_train=True, config=config)
    val_seq = BeatmapSequence(df=val, is_train=False, config=config)
    test_seq = BeatmapSequence(df=test, is_train=False, config=config)

    model = get_architecture_fn(config)(train_seq, False, config)

    callbacks = create_callbacks(train_seq, config)

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

    tf.keras.backend.clear_session()  # TF slowly leaks memory
    del train_seq, val_seq, test_seq
    gc.collect()

    return_list[:] = (history.history, eval_metrics)


def init_test():
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
    config.dataset.storage_folder = base_folder / 'new_datasets_config_test'
    # config.dataset.storage_folder = base_folder / 'test_datasets'
    # config.audio_processing.use_cache = False
    # generate_datasets(song_folders, config, prefix)
    train, val, test = load_datasets(config)

    timer('Loaded datasets', 0)
    # Ensure this song is excluded from the training data for hand tasting
    train.drop(index='133b', inplace=True, errors='ignore')
    train.drop(index='Daddy - PSY', inplace=True, errors='ignore')
    # dataset_stats(train)
    manager = multiprocessing.Manager()
    return_list = manager.list()
    return base_folder, return_list, test, timer, train, val
