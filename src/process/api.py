import multiprocessing
import os
from typing import Tuple
from time import time

import numpy as np
import pandas as pd

from process.compute import JSON, create_info, process_song_folder, path2mfcc_df
from utils.types import Config, Timer


def df2beatmap(df: pd.DataFrame, bpm: int = 60, events: Tuple = ()) -> Tuple[JSON, JSON]:
    beatmap = {
        '_notes': [],
        '_events': events,
    }

    for type_num, hand in [[0, 'l'], [1, 'r']]:
        cols = [x for x in df.columns if x[0] == hand]

        df_t = pd.DataFrame()
        df_t['_time'] = df.index
        df_t['_type'] = type_num

        for col in cols:
            df_t[col] = np.argmax(np.array(df[col].to_list()), axis=1)

        df_t = df_t.rename(columns={x: x[1:] for x in cols})

        beatmap['_notes'] += df_t.to_dict('records')

    info = create_info(bpm)

    return beatmap, info


def create_song_list(path):
    songs = []
    indicators = {'info.dat', 'info.json'}
    for root, _, files in os.walk(path, topdown=False):
        if bool(indicators.intersection(set(files))):
            songs.append(root)

    return songs


def songs2dataset(song_folders, config: Config):
    print(f'\tCreate dataframe from songs in folders: {len(song_folders):7} folders')
    timer = Timer()
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

    df = pd.concat(songs)
    timer('Concatenated songs')

    df = df.groupby(['name', 'difficulty']).apply(lambda x: generate_snippets(x, config=Config()))
    timer('Snippets generated')
    return df


def generate_snippets(song_df: pd.DataFrame, config: Config):
    stack = []
    ln = len(song_df)
    window = config.beat_preprocessing['snippet_window_length']
    skip = config.beat_preprocessing['snippet_window_skip']

    # Check if at least 1 window is possible
    if ln < window:
        return None

    # Name and difficulty information is contained in the grouping operation
    indexes_to_drop = ['name', 'difficulty']
    song_df = song_df.reset_index(level=indexes_to_drop).drop(columns=indexes_to_drop)

    for s in range(0, ln, skip):
        # Make sure the dataset contains ends of the songs
        if s + window > ln:
            stack.append(song_df.iloc[-window:])
        else:
            stack.append(song_df.iloc[s:s + window])

    df = pd.concat(stack, keys=list(range(0, len(song_df), skip)), names=['snippet', 'time'])
    return df


if __name__ == '__main__':
    song_folders = create_song_list('../data')
    total = len(song_folders)
    val_split = int(total * 0.8)
    test_split = int(total * 0.9)
    #

    result_path = '../data/test_beatmaps.pkl'
    df = pd.read_pickle(result_path)

    # res = df.groupby(['name', 'difficulty']).apply(lambda x: generate_snippets(x, config=Config()))

    print(df)

    # start = time()
    # df = songs2dataset(song_folders[test_split:], config=Config())
    # print(f'\n\nTook {time() - start}\n')
    #
    # df.to_pickle(result_path)
    # print(df)
    #
    # df = songs2dataset(song_folders[val_split:test_split], config=Config())
    # result_path = '../data/val_plain_beatmaps.pkl'
    # df.to_pickle(result_path)
    # print(df)

    df = songs2dataset(song_folders[:val_split], config=Config())
    result_path = '../data/train_plain_beatmaps.pkl'
    df.to_pickle(result_path)
    print(df)
    # pass

    # folder = '../data/new_dataformat/3207'
    # # df = path2mfcc_df(folder, Config())
    # # print(df)
    # df = process_song_folder(folder, Config())
    # print(df)
