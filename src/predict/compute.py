from time import time

import numpy as np
import pandas as pd
from tensorflow_core.python.keras import Model

from train.sequence import BeatmapSequence
from utils.types import Config


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
    most_recent = {name: val[:, 0:1] for name, val in data.items()}
    output_names = [f'prev_{name}' for name in stateful_model.output_names]

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
    for col, val in zip(output_names, pred):
        chose_index = np.random.choice(np.arange(val.shape[-1]), p=val.flatten())
        one_hot = np.zeros_like(val)
        one_hot[:, :, chose_index] = 1
        data[col][:, i + 1] = one_hot
        most_recent[col] = data[col][:, i + 1:i + 2]