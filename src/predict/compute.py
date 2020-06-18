import json
from functools import reduce
from pathlib import Path
from shutil import copy
from time import time
from typing import Dict, Tuple
from zipfile import ZipFile

import numba
import numpy as np
import pandas as pd
from tensorflow.keras import Model

from process.compute import process_song_folder
from train.sequence import BeatmapSequence
from utils.types import Config, JSON, Timer


def create_info(bpm):
    info = {
        "_version": "2.0.0",
        "_songName": "unknown",
        "_songSubName": "",
        "_songAuthorName": "unknown",
        "_levelAuthorName": "DeepSaber",
        "_beatsPerMinute": bpm,
        "_songTimeOffset": 0,
        "_shuffle": 0,
        "_shufflePeriod": 0.5,
        "_previewStartTime": 58.67857360839844,
        "_previewDuration": 10,
        "_environmentName": "BigMirrorEnvironment",
        "_customData": {
            "_contributors": [],
            "_customEnvironment": "",
            "_customEnvironmentHash": ""
        }
    }
    return info


def generate_beatmap(seq: BeatmapSequence, stateful_model: Model, config: Config):
    data = seq.data
    most_recent = {col: seq.data[col][:, 0:1] for col in stateful_model.input_names}  # initial beat
    output_names = [f'prev_{name}' for name in stateful_model.output_names]     # For TF 2.1 compatibility

    start = time()
    for i in range(len(seq.df) - 1):
        print(f'\r{i:4}: {time() - start:9.2}', end='', flush=True)
        pred = stateful_model.predict(most_recent)
        update_next(i, output_names, pred, data, most_recent)

    beatmap_df = predictions2df(data, seq)
    beatmap_df = append_last_prediction(beatmap_df, most_recent)

    for col in stateful_model.output_names:
        beatmap_df[col] = beatmap_df[f'prev_{col}']

    return beatmap_df


def append_last_prediction(beatmap_df, most_recent):
    last_row = beatmap_df.iloc[-1]
    last_time = float(last_row['next']) + beatmap_df.index[-1]
    added_row = pd.Series(most_recent, name=last_time).map(np.ndarray.flatten)
    beatmap_df = beatmap_df.append(added_row, ).fillna(method='ffill')
    return beatmap_df


def predictions2df(data, seq):
    beatmap_df = seq.df
    for col, val in data.items():
        beatmap_df[col] = np.split(val.flatten(), val.shape[1])
    beatmap_df = beatmap_df.reset_index('name').drop(columns='name')
    return beatmap_df


# @numba.njit()
def update_next(i, output_names, pred, data, most_recent):
    # for col, val in zip(output_names, pred):  # TF 2.1
    for col, val in pred.items():  # TF 2.2+
        col = f'prev_{col}'
        val = val ** 2
        chose_index = np.random.choice(np.arange(val.shape[-1]), p=val.flatten() / np.sum(val))
        one_hot = np.zeros_like(val)
        one_hot[:, :, chose_index] = 1
        data[col][:, i + 1] = one_hot
        most_recent[col] = data[col][:, i + 1:i + 2]


def zip_folder(folder_path):
    print(folder_path)
    with ZipFile((folder_path / folder_path.name).with_suffix('.zip'), 'w') as write_zip:
        for file in folder_path.glob('*.*'):
            if file.is_file() and file.suffix != '.zip':
                write_zip.write(file, file.name)


def update_generated_metadata(gen_folder, beatmap_folder, config):
    with open(beatmap_folder / 'info.dat', 'r') as rf:
        info = json.load(rf)
        difficulties = info['_difficultyBeatmapSets'][0]['_difficultyBeatmaps']
        info['_difficultyBeatmapSets'][0]['_difficultyBeatmaps'] = [x for x in difficulties if x['_difficulty']
                                                                    in config.training.use_difficulties]
        for not_generated in (x['_difficulty'] for x in difficulties
                              if x['_difficulty'] not in config.training.use_difficulties):
            (gen_folder / not_generated).with_suffix('.dat').unlink()
        info['_beatsPerMinute'] = 60

        with open(gen_folder / 'info.dat', 'w') as wf:
            json.dump(info, wf)


def save_generated_beatmaps(gen_folder, beatmap_dfs, config):
    for difficulty, df in beatmap_dfs.items():
        beatmap = df2beatmap(df, config)
        with open(gen_folder / f'{difficulty}.dat', 'w') as wf:
            json.dump(beatmap, wf)


def copy_folder_contents(in_folder, out_folder):
    for file in in_folder.glob('*.*'):
        if file.is_file():
            copy(file, out_folder)


def create_beatmap_dfs(stateful_model: Model, path: Path, config: Config) -> Dict[str, pd.DataFrame]:
    df = process_song_folder(str(path), config)

    config.beat_preprocessing.snippet_window_length = len(df)
    config.training.batch_size = 1
    # stateful_model = None
    output = {}

    for difficulty, sub_df in df.groupby('difficulty'):
        if difficulty not in config.training.use_difficulties:
            continue
        print(f'\nGenerating {difficulty}')
        seq = BeatmapSequence(sub_df.copy(), config)

        beatmap_df = generate_beatmap(seq, stateful_model, config)
        stateful_model.reset_states()

        output[difficulty] = beatmap_df
    return output


def df2beatmap(df: pd.DataFrame, config: Config, bpm: int = 60, events: Tuple = ()) -> JSON:
    beatmap = {
        '_version': '2.0.0',
        '_BPMChanges': [],
        '_notes': [],
        '_events': events,
    }

    plain_col_names = [x[2:] for x in config.dataset.beat_elements if x[0] == 'l' and 'cutDirection' not in x]
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
