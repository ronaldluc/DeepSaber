import random
from copy import deepcopy

import random
from copy import deepcopy
from multiprocessing import Pool
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from bayes_opt import JSONLogger, Events
from bayes_opt.util import load_logs
from tensorflow import keras

from predict.api import generate_complete_beatmaps
from process.api import load_datasets, create_song_list, songs2dataset
from train.metrics import Perplexity
from utils.types import Config, Timer


def main():
    timer = Timer()

    seed = 43  # random, non-fine tuned seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    config = Config()

    base_folder = config.base_data_folder

    config.dataset.storage_folder = base_folder / 'new_datasets_config_test'
    # config.dataset.storage_folder = base_folder / 'old_datasets'
    # config.dataset.storage_folder = base_folder / 'new_datasets'
    config.dataset.storage_folder = base_folder / 'test_datasets'
    config.audio_processing.use_cache = True

    # keras.mixed_precision.experimental.set_policy('mixed_float16')
    model_path = base_folder / 'temp'
    model_path.mkdir(parents=True, exist_ok=True)

    stateful_model = keras.models.load_model(model_path / 'stateful_model.keras',
                                             custom_objects={'Perplexity': Perplexity})
    stateful_model.summary()
    timer('Loaded stateful model', 5)

    storage_folder = base_folder / 'new_datasets'
    train, val, test = load_datasets(storage_folder)
    train_vec = get_vec_df(train)  # use train dataset to find good temperature
    # val_vec = get_vec_df(val)
    folder_name = 'deepsaber'
    config.dataset.beat_maps_folder = base_folder / 'evaluation_dataset' / folder_name
    config.dataset.storage_folder = base_folder / f'{folder_name}_datasets'

    human_velocities = compute_multiple_velocities(train_vec.iloc[:100000])

    # input_folder = base_folder / 'evaluation_dataset' / 'beat_sage'
    input_folder = base_folder / 'human_beatmaps' / 'new_dataformat'
    # output_folder = base_folder / 'testing' / 'generated_songs'
    output_folder = base_folder / 'evaluation_dataset' / 'deepsaber'

    timer = Timer()
    dirs = [x for x in input_folder.glob('*/') if x.is_dir()]
    dirs = list(x for x in test.index.to_frame()["name"].unique()[:13])
    print(dirs)

    def evaluate_temperature(temperature):
        config.generation.temperature = temperature
        for song_code in dirs:
            beatmap_folder = input_folder / song_code
            print(beatmap_folder)
            generate_complete_beatmaps(beatmap_folder, output_folder, stateful_model, deepcopy(config))
        timer('Generated beatmaps for all songs', 5)

        generated_velocities = velocities_from_config(deepcopy(config))
        distance = compute_avd_distance(human_velocities, generated_velocities)
        print(distance)
        return 1 - distance

    from bayes_opt import BayesianOptimization

    # Bounded region of parameter space
    pbounds = {'temperature': (0.0, 1.1)}

    optimizer = BayesianOptimization(
        f=evaluate_temperature,
        pbounds=pbounds,
        random_state=43,
    )

    logging_path = base_folder / 'logs/temperature_log_vec:vec.json'
    if logging_path.exists():
        load_logs(optimizer, logs=str(logging_path))
    logger = JSONLogger(path=str(logging_path))
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        # init_points=3,
        n_iter=25,
    )

    print(optimizer.max)


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
    params = [(df, window) for window in range(from_window_size, to_window_size)]

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
    df_vec = get_vec_df(df)
    generated_velocities_ = compute_multiple_velocities(df_vec)
    return generated_velocities_


if __name__ == '__main__':
    main()
