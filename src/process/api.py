import multiprocessing
import os
from typing import Tuple

import gensim
import numpy as np
import pandas as pd

from process.compute import create_ogg_paths, generate_snippets, \
    add_previous_prediction  # split needed for gColab upload
from process.compute import process_song_folder, create_ogg_caches, remove_ogg_cache
from utils.functions import create_word_mapping, check_consistency
from utils.types import Config, Timer


def create_song_list(path):
    songs = []
    indicators = {'info.dat', 'info.json'}
    for root, _, files in os.walk(path, topdown=False):
        if bool(indicators.intersection(set(files))):
            songs.append(root)

    return songs


def recalculate_mfcc_df_cache(song_folders, config: Config):
    """
    MFCC computation is memory heavy.
    Therefore recalculation catches SIGTERM through `multiprocessing`
    """
    if config.audio_processing.use_cache:
        return

    ogg_paths = create_ogg_paths(song_folders)
    remove_ogg_cache(ogg_paths)
    create_ogg_caches(ogg_paths, config)


def songs2dataset(song_folders, config: Config) -> pd.DataFrame:
    print(f'\tCreate dataframe from songs in folders: {len(song_folders):7} folders')
    timer = Timer()
    recalculate_mfcc_df_cache(song_folders, config)
    timer('Recalculated MFCC cache')

    pool = multiprocessing.Pool()
    folders_to_process = len(song_folders)

    inputs = ((s, config, (i, folders_to_process)) for i, s in enumerate(song_folders))
    songs = pool.starmap(process_song_folder, inputs)
    # songs = map(lambda x: process_song_folder(*x), inputs)    # single core version for debugging
    timer('Computed partial dataframes from folders')

    pool.close()
    pool.join()
    timer('Pool closed')

    songs = [x for x in songs if x is not None]
    timer('Filtered failed songs')

    if len(songs) == 0:
        raise ValueError(f'Dataset creation collected 0 songs. Check if searching in correct folders.')
    df = pd.concat(songs)
    timer('Concatenated songs')

    df = df_post_processing(df, config)

    df = df.groupby(['name', 'difficulty']).apply(lambda x: generate_snippets(x, config=config))
    timer('Snippets generated')
    return df


def df_post_processing(df, config):
    if config.dataset.action_word_model_path.exists():
        action_model = gensim.models.KeyedVectors.load(str(config.dataset.action_word_model_path))

        df['word_vec'] = np.vsplit(action_model[df['word'].values].astype('float16'), len(df))
        df['word_vec'] = df['word_vec'].map(lambda x: x[0])

        word_id_dict = create_word_mapping(action_model)
        df['word_id'] = df['word'].map(lambda word: word_id_dict.get(word, 1))  # 0: MASK, 1: UNK
    else:
        df['word_vec'] = 0
        df['word_id'] = 0

    df = df.groupby(['name', 'difficulty']).apply(lambda x: add_previous_prediction(x, config=config))

    return df

# if __name__ == '__main__':
#     song_folders = create_song_list('../data')
#     total = len(song_folders)
#     val_split = int(total * 0.8)
#     test_split = int(total * 0.9)
#     #
#
#     result_path = '../data/test_beatmaps.pkl'
#     df = pd.read_pickle(result_path)
#
#     # res = df.groupby(['name', 'difficulty']).apply(lambda x: generate_snippets(x, config=Config()))
#
#     print(df)
#
#     # start = time()
#     # df = songs2dataset(song_folders[test_split:], config=Config())
#     # print(f'\n\nTook {time() - start}\n')
#     #
#     # df.to_pickle(result_path)
#     # print(df)
#     #
#     # df = songs2dataset(song_folders[val_split:test_split], config=Config())
#     # result_path = '../data/val_plain_beatmaps.pkl'
#     # df.to_pickle(result_path)
#     # print(df)
#
#     df = songs2dataset(song_folders[:val_split], config=Config())
#     result_path = '../data/train_plain_beatmaps.pkl'
#     df.to_pickle(result_path)
#     print(df)
#     # pass
#
#     # folder = '../data/new_dataformat/3207'
#     # # df = path2mfcc_df(folder, Config())
#     # # print(df)
#     # df = process_song_folder(folder, Config())
#     # print(df)
def generate_datasets(song_folders, config: Config):
    timer = Timer()
    for phase, split in zip(['train', 'val', 'test'],
                            zip(config.training.data_split,
                                config.training.data_split[1:])
                            ):
        print('\n', '=' * 100, sep='')
        print(f'Processing {phase}')
        total = len(song_folders)
        split_from = int(total * split[0])
        split_to = int(total * split[1])
        result_path = config.dataset.storage_folder / f'{phase}_beatmaps.pkl'

        df = songs2dataset(song_folders[split_from:split_to], config=config)
        timer(f'Created {phase} dataset', 1)

        check_consistency(df)

        config.dataset.storage_folder.mkdir(parents=True, exist_ok=True)
        df.to_pickle(result_path, protocol=4)  # Protocol 4 for Python 3.6/3.7 compatibility
        timer(f'Pickled {phase} dataset', 1)


def load_datasets(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return [pd.read_pickle(config.dataset.storage_folder / f'{phase}_beatmaps.pkl') for phase in
            ['train', 'val', 'test']]
