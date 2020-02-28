import json
import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tensorflow import keras

from predict.api import create_beatmap_df, df2beatmap
from process.api import create_song_list, songs2dataset
from process.compute import process_song_folder
from train.callbacks import create_callbacks
from train.model import create_model
from train.sequence import BeatmapSequence, OnEpochEnd
from utils.functions import check_consistency
from utils.types import Config, Timer


def generate_datasets(config: Config):
    timer = Timer()
    for phase, split in zip(['train', 'val', 'test'],
                            zip(config.training['data_split'],
                                config.training['data_split'][1:])
                            ):
        print('\n', '=' * 100, sep='')
        print(f'Processing {phase}')
        split_from = int(total * split[0])
        split_to = int(total * split[1])
        result_path = config.dataset['storage_folder'] / f'{phase}_beatmaps.pkl'

        df = songs2dataset(song_folders[split_from:split_to], config=config)
        timer(f'Created {phase} dataset', 1)

        check_consistency(df)

        config.dataset['storage_folder'].mkdir(parents=True, exist_ok=True)
        df.to_pickle(result_path)
        timer(f'Pickled {phase} dataset', 1)


def load_datasets(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return [pd.read_pickle(os.path.join(config.dataset['storage_folder'], f'{phase}_beatmaps.pkl')) for phase in
            ['train', 'val', 'test']]


def dataset_stats(df: pd.DataFrame):
    print(df)
    print(df.groupby(['name', 'difficulty', 'snippet', 'time', ]).ngroups)
    print(df.groupby(['name', 'difficulty', 'snippet', ]).ngroups)
    print(df.groupby(['name', 'difficulty', ]).ngroups)
    print(df.groupby(['name', ]).ngroups)


def list2numpy(batch, col_name, groupby=('name')):
    return np.array(batch.groupby(list(groupby))[col_name].apply(list).to_list())


def create_training_data(X, groupby, config: Config):
    X_cols = config.dataset['audio']
    y_cols = config.dataset['beat_elements']
    return [list2numpy(X, col, groupby) for col in X_cols], \
           [list2numpy(X, col, groupby) for col in y_cols]


if __name__ == '__main__':
    # song_folders = create_song_list('../data')
    # total = len(song_folders)
    # print(f'Found {total} folders')
    #
    config = Config()
    # config.audio_processing['use_cache'] = False

    # generate_datasets(config)

    train, val, test = load_datasets(config)
    dataset_stats(train)

    train_seq = BeatmapSequence(train, config)
    val_seq = BeatmapSequence(val, config)
    test_seq = BeatmapSequence(test, config)

    model = create_model(train_seq, False, config)

    callbacks = create_callbacks(train_seq, config)

    timer = Timer()
    # model.fit(train_seq,
    #           validation_data=val_seq,
    #           callbacks=callbacks,
    #           epochs=16)
    timer('Training ')
    #
    model.save('../data/temp/model.keras')
    model = keras.models.load_model('../data/temp/model.keras')
    print('Evaluation')

    # gen_new_beat_map_path = song_folders[-2]
    # gen_new_beat_map_path = Path('../data/new_dataformat/4ede/')
    gen_new_beat_map_path = Path('../data/testing/generation/')
    gen_new_beat_map_path = Path('../data/testing/generation_normal/')
    #
    beatmap_df = create_beatmap_df(model, gen_new_beat_map_path, config)

    beatmap = df2beatmap(beatmap_df, config)

    gen_folder = gen_new_beat_map_path/'generated'
    gen_folder.mkdir(parents=True, exist_ok=True)

    difficulty = 'Hard'
    with open(gen_folder/f'{difficulty}.dat', 'w') as f:
        json.dump(beatmap, f)
    with open(gen_new_beat_map_path/'info.dat', 'r') as f:
        info = json.load(f)
        difficulties = info['_difficultyBeatmapSets'][0]['_difficultyBeatmaps']
        info['_difficultyBeatmapSets'][0]['_difficultyBeatmaps'] = [x for x in difficulties
                                                                    if x['_difficulty'] == difficulty]
        info['_beatsPerMinute'] = 60

        with open(gen_folder / 'info.dat', 'w') as new_f:
            json.dump(info, new_f)
