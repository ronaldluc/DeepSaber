import pandas as pd

from process.api import create_song_list, songs2dataset
from process.compute import path2beat_df
from utils.types import Config

if __name__ == '__main__':
    song_folders = create_song_list('../data')
    total = len(song_folders)
    val_split = int(total * 0.8)
    test_split = int(total * 0.9)

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