import multiprocessing
import os

import gensim
import pandas as pd
import numpy as np

from process.compute import create_ogg_paths, generate_snippets, \
    add_previous_prediction  # split needed for gColab upload
from process.compute import process_song_folder, create_ogg_caches, remove_ogg_cache
from utils.functions import create_word_mapping
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
    # songs = map(lambda x: process_song_folder(*x), inputs)
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

    action_model = gensim.models.KeyedVectors.load(str(config.dataset.action_word_model_path))
    timer('Loaded action model')

    df['word_vec'] = np.vsplit(action_model[df['word'].values].astype('float16'), len(df))
    df['word_vec'] = df['word_vec'].map(lambda x: x[0])
    timer('Generated action vectors')

    word_id_dict = create_word_mapping(action_model)
    df['word_id'] = df['word'].map(lambda word: word_id_dict.get(word, 1))
    timer('Generated action ids')

    df = df.groupby(['name', 'difficulty']).apply(lambda x: add_previous_prediction(x, config=config))
    timer('Added previous predictions')

    df = df.groupby(['name', 'difficulty']).apply(lambda x: generate_snippets(x, config=config))
    timer('Snippets generated')
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
