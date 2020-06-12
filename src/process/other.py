import random

from process.api import df2beatmap
from process.compute import beatmap2beat_df, path2beat_df, process_song_folder
from utils.functions import check_consistency
from utils.types import Config


def check_identity(df1):
    beatmap, info = df2beatmap(df1)
    df2 = beatmap2beat_df(beatmap, info)
    df3, df4 = df2.reset_index(), df1.reset_index()
    df3['_time'] = df3.index
    df4['_time'] = df4.index
    cols_to_round = ['_time', 'prev', 'next', 'part']
    df3[cols_to_round] = round(df3[cols_to_round])
    df4[cols_to_round] = round(df4[cols_to_round])
    for col in df3.columns:
        print(col, (df3[col] == df4[col]).all())


def sanity_check():
    # Check whether the functions perform reverse operations
    print('\nTesting old format')
    df1 = path2beat_df('../data/old_dataformat/AMENOHOAKARI/ExpertPlus.json',
                       '../data/old_dataformat/AMENOHOAKARI/info.json')
    check_identity(df1)
    print('\nTesting new format')
    df1 = path2beat_df('../data/new_dataformat/3aa4/ExpertPlus.dat',
                       '../data/new_dataformat/3aa4/info.dat')
    check_identity(df1)

    config = Config()
    # config.audio_processing.use_cache = False
    df1 = process_song_folder('../data/new_dataformat/3207', config=config)

    print(check_consistency(df1))
    df1['l_lineLayer'] = df1['l_lineLayer'].map(lambda x: [0.0] * random.randint(3, 5))
    print(check_consistency(df1))

# if __name__ == '__main__':
#     sanity_check()
