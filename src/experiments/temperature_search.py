import random
import shutil
from copy import deepcopy
from multiprocessing import Pool
from typing import Tuple, Optional

import kerastuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from bayes_opt import BayesianOptimization
from bayes_opt import JSONLogger, Events
from bayes_opt.util import load_logs
from tensorflow import keras

from experiments.compute import init_test
from predict.api import generate_complete_beatmaps
from process.api import load_datasets, create_song_list, songs2dataset
from train.callbacks import create_callbacks
from train.metrics import Perplexity
from train.model import get_architecture_fn, save_model
from train.sequence import BeatmapSequence
from utils.types import Config, Timer, ModelType, DatasetConfig


def main():
    base_folder, return_list, test, timer, train, val = init_test()

    seed = 43  # random, non-fine tuned seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    config = Config()

    config.dataset.storage_folder = base_folder / 'new_datasets'
    config.audio_processing.use_cache = True

    model_path = base_folder / 'temp'
    model_path.mkdir(parents=True, exist_ok=True)

    configuration_name = 'MLSTM'
    config.training.model_type = ModelType.TUNE_MLSTM
    config.training.batch_size = 128
    config.training.label_smoothing = 0.5
    config.training.mixup_alpha = 0.5
    config.training.x_groups = [['prev_word_id', 'prev_word_vec'], DatasetConfig().categorical,
                                DatasetConfig().audio, DatasetConfig().regression]
    config.training.y_groups = [['word_id'], ]
    hp = kt.HyperParameters()
    fixed_params = {'connections_0': 2,
                    'connections_1': 2,
                    'connections_2': 2,
                    'connections_3': 3,
                    'connections_4': 1,
                    'connections_5': 3,
                    'connections_6': 2,
                    'depth_0': 18,
                    'depth_1': 23,
                    'depth_2': 43,
                    'depth_3': 13,
                    'depth_4': 52,
                    'depth_5': 5,
                    'depth_6': 11,
                    'dropout_0': 0.25612932926324405,
                    'dropout_1': 0.1620424523625309,
                    'dropout_2': 0.4720468723284278,
                    'dropout_3': 0.43881829788147036,
                    'dropout_4': 0.44741780640383355,
                    'dropout_5': 0.3327191857714107,
                    'dropout_6': 0.1367707920005909,
                    'initial_learning_rate': 0.008,
                    'label_smoothing': 0.13716631669361445,
                    'lstm_layers': 3,
                    'width_0': 16,
                    'width_1': 9,
                    'width_2': 15,
                    'width_3': 16,
                    'width_4': 5,
                    'width_5': 11,
                    'width_6': 4,
                    }
    for param, value in fixed_params.items():
        hp.Fixed(param, value=value)
    find_temperature_and_generate(base_folder, train, val, test, model_path, configuration_name, deepcopy(config), hp)

    configuration_name = 'id:id'
    config.training.model_type = ModelType.CUSTOM
    config.training.cnn_repetition = 2
    config.training.lstm_repetition = 2
    config.training.dense_repetition = 0
    config.training.model_size = 512
    config.training.dropout = 0.4
    config.training.initial_learning_rate = 1e-2
    config.training.batch_size = 128
    config.training.label_smoothing = 0.5
    config.training.mixup_alpha = 0.5
    config.training.l2_regularization = 0
    config.training.x_groups = [['prev_word_id', ], DatasetConfig().categorical,
                                DatasetConfig().audio, DatasetConfig().regression]
    config.training.y_groups = [['word_id'], ]
    find_temperature_and_generate(base_folder, train, val, test, model_path, configuration_name, deepcopy(config))

    configuration_name = 'vec:vec'
    config.training.model_type = ModelType.CUSTOM
    config.training.cnn_repetition = 2
    config.training.lstm_repetition = 2
    config.training.dense_repetition = 0
    config.training.model_size = 512
    config.training.dropout = 0.4
    config.training.initial_learning_rate = 1e-2
    config.training.batch_size = 128
    config.training.label_smoothing = 0.5
    config.training.mixup_alpha = 0.5
    config.training.l2_regularization = 0
    config.training.x_groups = [['prev_word_vec', ], DatasetConfig().categorical,
                                DatasetConfig().audio, DatasetConfig().regression]
    config.training.y_groups = [['word_vec'], ]
    find_temperature_and_generate(base_folder, train, val, test, model_path, configuration_name, deepcopy(config))


def find_temperature_and_generate(base_folder, train, val, test, model_path, test_name, config,
                                  hp: Optional[kt.HyperParameters] = None):
    timer = Timer()

    train_seq = BeatmapSequence(df=train, is_train=True, config=config)
    val_seq = BeatmapSequence(df=val, is_train=False, config=config)
    test_seq = BeatmapSequence(df=test, is_train=False, config=config)

    model = get_architecture_fn(config)(train_seq, False, config)
    if hp is not None:
        model = model(hp, use_avs_model=True)
    model.summary()
    callbacks = create_callbacks(train_seq, config)
    model.fit(train_seq,
              validation_data=val_seq,
              callbacks=callbacks,
              epochs=400,
              verbose=2,
              workers=10,
              max_queue_size=16,
              use_multiprocessing=False,
              )
    timer('Trained model', 5)
    model.evaluate(test_seq)
    timer('Evaluated model', 5)
    save_model(model, model_path, train_seq, config, hp)
    timer('Saved model', 5)

    stateful_model = keras.models.load_model(model_path / 'stateful_model.keras',
                                             custom_objects={'Perplexity': Perplexity, 'mish': tfa.activations.mish})
    stateful_model.summary()
    timer('Loaded stateful model', 5)
    storage_folder = base_folder / 'new_datasets'
    train, val, test = load_datasets(storage_folder)
    train_vec = get_vec_df(train)  # use train dataset to find good temperature
    folder_name = 'deepsaber'
    config.dataset.beat_maps_folder = base_folder / 'testing' / 'generated_songs'
    config.dataset.storage_folder = base_folder / f'{folder_name}_datasets'
    human_velocities = compute_multiple_velocities(train_vec.iloc[:100000])
    input_folder = base_folder / 'human_beatmaps' / 'new_dataformat'
    output_folder = config.dataset.beat_maps_folder
    timer = Timer()
    dirs = list(x for x in test.index.to_frame()["name"].unique()[:10])
    print(dirs)

    def evaluate_temperature(temperature):
        for out_file in output_folder.glob('*'):
            if out_file.is_dir():
                shutil.rmtree(out_file)

        config.generation.temperature = temperature
        for song_code in dirs:
            beatmap_folder = input_folder / song_code
            print(beatmap_folder)
            generate_complete_beatmaps(beatmap_folder, output_folder, stateful_model, deepcopy(config))
        timer('Generated beatmaps for all songs', 5)

        generated_velocities = velocities_from_config(deepcopy(config))
        distance = compute_avd_distance(human_velocities, generated_velocities)
        return -distance  # we need to maximize

    pbounds = {'temperature': (0.7, 3.0)}
    optimizer = BayesianOptimization(
        f=evaluate_temperature,
        pbounds=pbounds,
        random_state=43,
    )
    logging_path = base_folder / f'logs/temperature_log_{test_name}.json'
    if logging_path.exists():
        optimizer = load_logs(optimizer, logs=str(logging_path))
    logger = JSONLogger(path=str(logging_path))
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(
        init_points=0,
        n_iter=1,
    )
    print(f'{optimizer.max=}')

    input_folder = base_folder / 'evaluation_dataset' / 'beat_sage_expert'
    output_folder = base_folder / 'evaluation_dataset' / f'deepsaber_{test_name}'
    dirs = [x for x in input_folder.glob('*/') if x.is_dir()]
    config.generation.temperature = optimizer.max['params']['temperature']

    for song_code in dirs:
        beatmap_folder = input_folder / song_code
        print(f'Working on {beatmap_folder.name}')
        generate_complete_beatmaps(beatmap_folder, output_folder, stateful_model, config)
        timer('Generated beatmaps', 5)


def load_datasets(storage_folder) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return [pd.read_pickle(storage_folder / f'{phase}_beatmaps.pkl') for phase in
            ['train', 'val', 'test']]


def get_vec_df(df):
    print(df.head(2))
    nodup = df.droplevel(2)
    nodup = nodup.loc[~nodup.index.duplicated()]

    top = 900000  # 9000
    return pd.DataFrame(np.array(nodup.word_vec.values.tolist())[:top], index=nodup.index[:top])


def cosine_dist(a, b):
    return 1 - np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def compute_velocity(df, window=7):
    means = df.iloc[::].rolling(window, win_type='boxcar').mean(std=7, tau=7, sym=False)

    #     velocity  = cosine_dist(means.values, means.shift(window).values)
    diff = means - means.shift(window)
    velocity = ((diff.dropna() ** 2).sum(axis=1)) ** (1 / 2)
    return pd.Series(velocity).dropna()


def compute_complete_velocity(ser, window):
    return window, ser.groupby(['name', 'difficulty']).apply(lambda ser_: compute_velocity(ser_, window))


def compute_multiple_velocities(df, from_window_size=1, to_window_size=32):
    pool = Pool()
    params = [(df, window) for window in range(from_window_size, to_window_size + 1)]

    cached_train_v = dict(pool.starmap(compute_complete_velocity, params))
    #     Single-core version for laptops, etc.
    #     cached_train_v = dict(starmap(compute_complete_velocity, params))

    pool.close()
    pool.join()
    return cached_train_v


from scipy.stats import ks_2samp


def compute_ks_statistic(window1, data1, window2, data2):
    assert window1 == window2
    stat = ks_2samp(data1, data2)[0]
    return window1, stat


def compute_avd_distance(val_velocities, generated_velocities):
    pool = Pool()

    params = [(w1, d1, w2, d2) for (w1, d1), (w2, d2) in zip(val_velocities.items(), generated_velocities.items())]
    res = dict(pool.starmap(compute_ks_statistic, params))
    #     Single-core version for laptops, etc.
    #     res = dict(starmap(compute_ks_statistic, params))

    pool.close()
    pool.join()
    return pd.Series(res).mean()


def velocities_from_config(config: Config):
    song_folders = create_song_list(config.dataset.beat_maps_folder)
    df = songs2dataset(song_folders, config)
    print(df.head(2))
    df_vec = get_vec_df(df)
    generated_velocities_ = compute_multiple_velocities(df_vec)
    return generated_velocities_


if __name__ == '__main__':
    main()
