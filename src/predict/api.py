from functools import reduce
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model

from predict.compute import generate_beatmap
from process.compute import process_song_folder
from train.model import create_model
from train.sequence import BeatmapSequence
from utils.types import Config, JSON


def create_beatmap_df(model: Model, path: Path, config: Config) -> pd.DataFrame:
    df = process_song_folder(str(path), config)

    config.beat_preprocessing['snippet_window_length'] = len(df)
    config.training['batch_size'] = 1
    seq = BeatmapSequence(df, config)

    # stateful_model = model
    stateful_model = create_model(seq, True, config)

    stateful_model.set_weights(model.get_weights())

    beatmap_df = generate_beatmap(seq, stateful_model, config)

    path = '../data/temp/beatmap_df.pkl'
    beatmap_df.to_pickle(path)

    path = '../data/temp/beatmap_df.pkl'
    beatmap_df = pd.read_pickle(path)

    # use heat Done!
    return beatmap_df


def df2beatmap(df: pd.DataFrame, config: Config, bpm: int = 60, events: Tuple = ()) -> JSON:
    beatmap = {
        '_version': '2.0.0',
        '_BPMChanges': [],
        '_notes': [],
        '_events': events,
    }

    plain_col_names = [x[2:] for x in config.dataset['beat_elements'] if x[0] is 'l' and 'cutDirection' not in x]
    partially_equal_beat_elements = [df[f'l_{col}'].map(np.ndarray.argmax)
                                     == df[f'r_{col}'].map(np.ndarray.argmax)
                                     for col in plain_col_names]
    equal_beat_elements = reduce(lambda x, y: x & y, partially_equal_beat_elements)

    df['equal_beat_elements'] = False
    df.loc[equal_beat_elements, 'equal_beat_elements'] = True
    df['even'] = False
    df.loc[::2, 'even'] = True

    df.loc[equal_beat_elements & df['even'], [f'r_{x}' for x in plain_col_names]] = np.nan
    df.loc[equal_beat_elements & ~df['even'], [f'l_{x}' for x in plain_col_names]] = np.nan

    for type_num, hand in [[0, 'l'], [1, 'r']]:
        cols = [x for x in df.columns if x[0] == hand]

        df_t = pd.DataFrame(index=df.index)
        df_t['_time'] = df.index
        df_t['_type'] = type_num

        df_t[cols] = df[cols]
        df_t = df_t.dropna()

        for col in cols:
            df_t[col] = np.argmax(np.array(df_t[col].to_list()), axis=1)

        df_t = df_t.rename(columns={x: x[1:] for x in cols})

        beatmap['_notes'] += df_t.to_dict('records')

    # data2JSON Done!
    return beatmap


# if __name__ == '__main__':
#     gen_new_beat_map_path = '../data/new_dataformat/4ede/'
#     config = Config()
#     #
#     # df1 = songs2dataset([gen_new_beat_map_path, ], config)
#     #
#     # df2 = process_song_folder(gen_new_beat_map_path, config)
#     # config.beat_preprocessing['snippet_window_length'] = len(df2)
#     #
#     # seq = BeatmapSequence(df2, config)
#     # # ['name', 'difficulty', 'snippet', 'time']
#     # print('done')
#
#     path = '../data/temp/beatmap_df.pkl'
#     df = pd.read_pickle(path)
#     df2beatmap(df, config)
#     print(df)
