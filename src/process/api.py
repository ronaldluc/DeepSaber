import multiprocessing
import os
from typing import Tuple

import numpy as np
import pandas as pd

from process.compute import JSON, create_info, process_song_folder, path2mfcc
from utils.types import Config


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


def songs2dataset(song_folders):
    pool = multiprocessing.Pool()
    songs = pool.map(process_song_folder, song_folders)
    # songs = map(process_song_folder, song_folders)
    pool.close()
    pool.join()

    songs = [x for x in songs if x is not None]
    df = pd.concat(songs)
    return df


if __name__ == '__main__':
    song_folders = create_song_list('../data')
    # total = len(song_folders)
    # val_split = int(total * 0.8)
    # test_split = int(total * 0.9)
    #
    # start = time()
    # df = songs2dataset(song_folders[test_split:])
    # print(f'\n\nTook {time() - start}\n')
    #
    # result_path = '../data/test_beatmaps.pkl'
    # df.to_pickle(result_path)
    # print(df)
    #
    # df = songs2dataset(song_folders[val_split:test_split])
    # result_path = '../data/val_plain_beatmaps.pkl'
    # df.to_pickle(result_path)
    # print(df)
    #
    # df = songs2dataset(song_folders[:val_split])
    # result_path = '../data/train_plain_beatmaps.pkl'
    # df.to_pickle(result_path)
    # print(df)
    # pass

    folder = '../data/new_dataformat/3207'
    df = path2mfcc(folder, Config())
    print(df)
    df = process_song_folder(folder)
    print(df)
