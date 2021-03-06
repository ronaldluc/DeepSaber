import json
from functools import reduce
from itertools import product
from pathlib import Path
from shutil import copy
from time import time
from typing import Dict, Tuple
from zipfile import ZipFile

import gensim
import numpy as np
import pandas as pd
from scipy.special import softmax
from tensorflow.keras import Model

from process.api import df_post_processing, normalize_columns
from process.compute import process_song_folder
from train.sequence import BeatmapSequence
from utils.types import Config, JSON


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


def generate_beatmap(beatmap_df: pd.DataFrame, seq: BeatmapSequence, stateful_model: Model,
                     action_model: gensim.models.KeyedVectors,
                     word_id_dict: Dict[str, int], config: Config):
    most_recent = {col: seq[0][0][col][:, 0:1] for col in stateful_model.input_names}  # initial beat
    output_names = [f'prev_{name}' for name in stateful_model.output_names]  # For TF 2.1 compatibility
    reverse_word_id_dict = {val: key for key, val in word_id_dict.items()}

    # Reset the whole seq.data columns except for the first action to prevent information leaking
    for col in product(['', 'prev_'], ['word_id', 'word_vec'] + config.dataset.beat_elements):
        seq.data[''.join(col)][:, 1:, :] = 0.0

    start = time()
    total_len = len(beatmap_df) - 1
    temperature = config.generation.temperature  # TODO: change toconfig.generation.temperature(0)
    for i in range(len(beatmap_df) - 1):
        elapsed = time() - start
        print(f'\r{i:4}: {int(elapsed):3} / ~{int(elapsed * total_len / (i + 1)):3} s', end='', flush=True)
        pred = stateful_model.predict(most_recent)

        # word_vec to word_id prob
        if 'word_vec' in stateful_model.output_names:
            closest_words = action_model.similar_by_vector(pred['word_vec'].flatten(), topn=30, restrict_vocab=None)

            pred['word_id'] = np.zeros((1, 1, config.dataset.num_classes['word_id']))
            for word, distance in closest_words:
                pred['word_id'][:, :, word_id_dict[word]] = distance

        update_next(i, pred, seq, temperature, config)

        update_action_representations(i, action_model, seq, word_id_dict, pred, reverse_word_id_dict, config)

        if set(stateful_model.output_names) >= set(config.dataset.beat_elements):
            clip_next_to_closest_existing(i, action_model, seq, word_id_dict, config)

        # get last action in the correct format
        most_recent = {col: seq[0][0][col][:, i + 1:i + 2] for col in stateful_model.input_names}

        # Experiment with moving temperature based on AVD distance. Needs further research
        # temperature = responsive_temperature(seq, temperature, i)

    save_velocity_hist(seq, config)
    beatmap_df = predictions2df(beatmap_df, seq)
    # beatmap_df = append_last_prediction(beatmap_df, most_recent)    # TODO: Remove if unnecessary

    for col in stateful_model.output_names:
        beatmap_df[col] = beatmap_df[f'prev_{col}']

    return beatmap_df[stateful_model.output_names]  # output only generated columns


def responsive_temperature(seq: BeatmapSequence, temperature, i):
    window_size = 9
    new = seq.data['prev_word_vec'][:, i - window_size + 1:i + 1].mean(axis=1)
    old = seq.data['prev_word_vec'][:, i - window_size - window_size // 2:i - window_size // 2 + 1].mean(axis=1)
    # print(f' {new.shape=} {old.shape=}', end='')
    if new.shape == old.shape:
        velocity = (np.sum((new - old) ** 2)) ** (1 / 2)
        if np.isfinite(velocity):
            temperature = min(0.95 - velocity * 0.05, 0.75)
        print(f' {velocity:4.2f} | {temperature:4.2f}', end='')
    return temperature


def cosine_dist(a, b):
    return 1 - np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def l2_dist(a, b):
    diff = a - b
    dist = ((diff.dropna() ** 2).sum(axis=1)) ** (1 / 2)
    return dist


def save_velocity_hist(seq: BeatmapSequence, config: Config):
    mean = pd.DataFrame(seq.data['prev_word_vec'][0]).rolling(7).mean()
    velocity = l2_dist(mean, mean.shift(4))
    # velocity = cosine_dist(mean.values, mean.shift(4).values)
    ax = pd.Series(velocity).dropna().plot.hist(bins=24, figsize=(14, 6), density=True, alpha=0.5, )
    ax.set_xlim(0, 5)
    fig = ax.get_figure()
    fig.savefig(config.base_data_folder / 'temp' / 'distribution.pdf')


def update_action_representations(i: int, action_model: gensim.models.KeyedVectors, seq: BeatmapSequence,
                                  word_id_dict: Dict[str, int], pred: Dict[str, np.ndarray],
                                  reverse_word_id_dict: Dict[int, np.ndarray], config: Config):
    # update all representations, to make interesting models possible without data leaking.
    if 'word_id' in pred.keys():  # `word_id` is the prefered action representation
        word_str = reverse_word_id_dict[int(seq.data['prev_word_id'][:, i + 1])]
        seq.data['prev_word_vec'][:, i + 1] = action_model[word_str]
        word_str2per_attribute(i, word_str, seq)
    elif 'word_vec' in pred.keys():
        closest_word_str = action_model.similar_by_vector(seq.data['prev_word_vec'][:, i + 1],
                                                          topn=1, restrict_vocab=config.generation.restrict_vocab)[0][0]
        seq.data['prev_word_id'][:, i + 1] = word_id_dict[closest_word_str]
        word_str2per_attribute(i, closest_word_str, seq)
    else:
        prev_word = per_attribute2word_str(i, seq)
        seq.data['prev_word_vec'][:, i + 1] = action_model[prev_word]
        closest_word_str = action_model.similar_by_vector(seq.data['prev_word_vec'][:, i + 1],
                                                          topn=1, restrict_vocab=config.generation.restrict_vocab)[0][0]
        seq.data['prev_word_id'][:, i + 1] = word_id_dict[closest_word_str]


def per_attribute2word_str(i: int, seq: BeatmapSequence):
    word = []
    for hand in 'lr':
        word += [hand.upper()]
        word += [np.argmax(seq.data[f'prev_{hand}_{name}'][:, i + 1], axis=-1).astype(str)[0] for name in
                 ['lineLayer', 'lineIndex', 'cutDirection']]
        word += ['_']
    prev_word = ''.join(word[:-1])
    return prev_word


def clip_next_to_closest_existing(i: int, action_model: gensim.models.KeyedVectors, seq: BeatmapSequence,
                                  word_id_dict: Dict[str, int], config: Config):
    prev_word = per_attribute2word_str(i, seq)
    closest_word_str = action_model.similar_by_vector(action_model[prev_word],
                                                      topn=1, restrict_vocab=config.generation.restrict_vocab)[0][0]
    seq.data['prev_word_id'][:, i + 1] = word_id_dict[closest_word_str]
    seq.data['prev_word_vec'][:, i + 1] = action_model[closest_word_str]
    closest_word_str = seq.data['prev_word']

    word_str2per_attribute(i, closest_word_str, seq)


def word_str2per_attribute(i: int, closest_word_str: str, seq: BeatmapSequence):
    action_dim_values = *closest_word_str[1:4], *closest_word_str[-3:]  # extract first and last beat elements
    for (hand, dim), chosen_index in zip(product('lr', ['lineLayer', 'lineIndex', 'cutDirection']),
                                         action_dim_values):
        col = f'prev_{hand}_{dim}'
        if closest_word_str == 'UNK' or closest_word_str == 'MASK':
            seq.data[col][:, i + 1] = seq.data[col][:, i]
        else:
            seq.data[col][:, i + 1] = chosen_index


def append_last_prediction(beatmap_df: pd.DataFrame, most_recent):
    last_row = beatmap_df.iloc[-1]
    last_time = float(last_row['next']) + beatmap_df.index[-1]
    added_row = pd.Series(most_recent, name=last_time).map(np.ndarray.flatten)
    beatmap_df = beatmap_df.append(added_row, ).fillna(method='ffill')
    return beatmap_df


def predictions2df(beatmap_df: pd.DataFrame, seq: BeatmapSequence):
    for col, val in seq.data.items():
        beatmap_df[col] = np.split(val.flatten(), val.shape[1])
    beatmap_df = beatmap_df.reset_index('name').drop(columns='name')
    return beatmap_df


# @numba.njit()
def update_next(i: int, pred: Dict[str, np.ndarray], seq: BeatmapSequence, temperature, config: Config):
    # for col, val in zip(output_names, pred):  # TF 2.1
    for col, val in pred.items():  # TF 2.2+
        col = f'prev_{col}'

        if col in seq.categorical_cols:
            # with np.errstate(divide='ignore'):
            val = np.log(val + 1e-9) / np.max([temperature, 1e-6])
            val = softmax(val, axis=-1)
            chosen_index = np.random.choice(np.arange(val.shape[-1]), p=val.flatten() / np.sum(val))  # categorical dist
            seq.data[col][:, i + 1] = chosen_index
        else:  # regression cols
            seq.data[col][:, i + 1] = val


def zip_folder(folder_path: Path):
    print(folder_path)
    with ZipFile((folder_path / folder_path.name).with_suffix('.zip'), 'w') as write_zip:
        for file in folder_path.glob('*.*'):
            if file.is_file() and file.suffix != '.zip':
                write_zip.write(file, file.name)


def update_generated_metadata(gen_folder: Path, beatmap_folder: Path, config: Config):
    info_file: Path = [x for x in beatmap_folder.glob('*.dat') if x.name.lower() == 'info.dat'][0]
    with open(info_file, 'r') as rf:
        info = json.load(rf)
        difficulties = info['_difficultyBeatmapSets'][0]['_difficultyBeatmaps']
        info['_difficultyBeatmapSets'][0]['_difficultyBeatmaps'] = [x for x in difficulties if x['_difficulty']
                                                                    in config.training.use_difficulties]
        for not_generated in (x['_difficulty'] for x in difficulties
                              if x['_difficulty'] not in config.training.use_difficulties):
            (gen_folder / not_generated).with_suffix('.dat').unlink(missing_ok=True)
        info['_beatsPerMinute'] = 60
        info['_levelAuthorName'] = 'DeepSaber'

        with open(gen_folder / 'info.dat', 'w') as wf:
            json.dump(info, wf)


def save_generated_beatmaps(gen_folder: Path, beatmap_dfs: Dict[str, pd.DataFrame],
                            action_model: gensim.models.KeyedVectors,
                            word_id_dict: Dict[str, int], config):
    for difficulty, df in beatmap_dfs.items():
        beatmap = df2beatmap(df, action_model, word_id_dict, config)
        with open(gen_folder / f'{difficulty}.dat', 'w') as wf:
            json.dump(beatmap, wf)


def copy_folder_contents(in_folder: Path, out_folder: Path):
    for file in in_folder.glob('*.*'):
        if file.is_file():
            copy(file, out_folder)


def create_beatmap_dfs(stateful_model: Model, action_model: gensim.models.KeyedVectors,
                       word_id_dict: Dict[str, int], path: Path, config: Config) -> Dict[str, pd.DataFrame]:
    df = process_song_folder(str(path), config)
    df = df_post_processing(df, config)
    df = normalize_columns(df, config)

    config.beat_preprocessing.snippet_window_length = len(df)
    config.training.batch_size = config.generation.batch_size
    output = {}

    for difficulty, sub_df in df.groupby('difficulty'):
        if difficulty not in config.training.use_difficulties:
            continue
        print(f'\n\tGenerating {difficulty}')
        seq = BeatmapSequence(df=sub_df, is_train=False, config=config)

        # beatmap_df = sub_df.copy()    # bypass the generation
        beatmap_df = generate_beatmap(sub_df.copy(), seq, stateful_model, action_model,
                                      word_id_dict, config)
        stateful_model.reset_states()

        output[difficulty] = beatmap_df
    return output


def df2beatmap(df: pd.DataFrame, action_model: gensim.models.KeyedVectors,
               word_id_dict: Dict[str, int], config: Config, bpm: int = 60, events: Tuple = ()) -> JSON:
    beatmap = {
        '_version': '2.0.0',
        '_BPMChanges': [],
        '_notes': [],
        '_events': events,
    }
    df.index = df.index.to_frame()['time']  # only time from the multiindex is needed
    inverse_word_id_dict = {val: key for key, val in word_id_dict.items()}
    if 'word_id' in df.columns:
        df['word_id'] = np.array(df['word_id'].to_list()).flatten()
        df = df.loc[df['word_id'] > 1]
        word = df['word_id'].map(lambda word_id: inverse_word_id_dict[word_id])
        beatmap['_notes'] += word_ser2json(word)
    elif 'word_vec' in df.columns:
        word = df['word_vec'].map(lambda vec:
                                  action_model.similar_by_vector(vec, topn=1,
                                                                 restrict_vocab=config.generation.restrict_vocab)[0][0])
        beatmap['_notes'] += word_ser2json(word)
    else:
        beatmap['_notes'] += double_beat_element2json(df, config)

    return beatmap


def double_beat_element2json(df: pd.DataFrame, config: Config):
    notes = []
    plain_col_names = [x[2:] for x in config.dataset.beat_elements if x[0] == 'l' and 'cutDirection' not in x]
    partially_equal_beat_elements = [df[f'l_{col}'].astype(int)
                                     == df[f'r_{col}'].astype(int)
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
        df_t['_time'] = df_t.index
        df_t['_type'] = type_num

        df_t[cols] = df[cols]
        df_t = df_t.dropna()

        for col in cols:
            df_t[col] = np.array(df_t[col].to_list()).flatten()

        df_t = df_t.rename(columns={x: x[1:] for x in cols})

        notes += df_t.to_dict('records')
    return notes


def word_ser2json(word: pd.Series) -> Dict:
    word = word.str.split('_').explode()
    df_t = pd.DataFrame(word.str.split('').tolist(),
                        columns=['drop', 'hand', '_lineLayer', '_lineIndex', '_cutDirection', 'drop'], )

    df_t['_type'] = 0
    df_t.loc[df_t['hand'] == 'R', '_type'] = 1
    df_t = df_t.drop(columns=['hand', 'drop'])
    df_t = df_t.astype(int)
    df_t['_time'] = word.index
    return df_t.to_dict('records')
