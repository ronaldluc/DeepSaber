import json
import os
from sys import stderr
from time import sleep
from typing import Union, Mapping, List

import numpy as np
import pandas as pd
import sonopy as sonopy
import soundfile as sf
import speechpy
import tensorflow as tf
import numba

from utils.functions import progress
from utils.types import Config, Timer

JSON = Union[str, int, float, bool, None, Mapping[str, 'JSON'], List['JSON']]


def to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn beat elements dimensions into one-hot-encoding
    :param df: beat elements
    :return: updated beat elements
    """
    dim_max = {'_lineLayer': 3, '_lineIndex': 4, '_cutDirection': 9}
    for col, dim in dim_max.items():
        df[col] = tf.keras.utils.to_categorical(df[col], dim).tolist()
    return df


def one_beat_element_per_hand(df: pd.DataFrame) -> pd.Series:
    """

    :param df: beat elements
    :return:
    """
    hands = []
    for hand in range(2):
        hands.append(df.loc[df['_type'] == hand][:1])

    # if only one hand has beat element, both predict the same
    for hand in range(2):
        if hands[hand].empty:
            hands[hand] = hands[hand - 1].copy()

    for hand in range(2):
        hands[hand] = hands[hand].iloc[0].drop(['_type', '_time'])

    return hands[0].add_prefix('l').append(hands[1].add_prefix('r'))


@numba.njit()
def compute_true_time(beat_elements: np.ndarray, bpm_changes: np.ndarray, start_bpm: float) -> np.ndarray:
    """
    Calculate beat elements times in seconds. Originally in beats since beginning of the song.
    :param beat_elements: times of beat elements, sorted
    :param bpm_changes: [time, bpm], sorted by time
    :param start_bpm: initial bpm
    :return: time of beat_elements in seconds
    """
    true_time = np.zeros_like(beat_elements, dtype=np.float_)

    current_bpm = start_bpm
    current_beat = 0.0  # Current time in beats since beginning
    current_time = 0.0  # Current time in seconds since beginning
    event_index = 0

    for i in range(beat_elements.shape[0]):
        # Apply BPM changes that happened between this and last beat element
        while event_index < bpm_changes.shape[0] and bpm_changes[event_index, 0] < beat_elements[i]:
            bpm_change = bpm_changes[event_index]
            current_time += (bpm_change[0] - current_beat) * (60.0 / current_bpm)
            current_beat = bpm_change[0]
            current_bpm = bpm_change[1]
            event_index += 1

        current_time += (beat_elements[i] - current_beat) * (60.0 / current_bpm)
        current_beat = beat_elements[i]
        true_time[i] = current_time
    return true_time


def compute_time_cols(df):
    """
    Compute `prev`, `next`, `part`.
    :param df: beat elements
    :return: beat elements
    """
    df['time'] = df.index
    # previous beat in seconds
    df['prev'] = df['time'].diff()
    df['prev'] = df['prev'].fillna(df['prev'].max())
    # next beat in seconds
    df['next'] = df['prev'].shift(periods=-1)
    df['next'] = df['next'].fillna(df['next'].max())
    # which part of the song each beat belongs to
    df['part'] = df['time'] / df['time'].max()
    df = df.drop(columns='time')

    return df


def create_bpm_df(beatmap: JSON) -> pd.DataFrame:
    """
    Create df with bpm changing event
    :param beatmap: json
    :return:
    """
    bpm_df = pd.DataFrame(
        beatmap['_events'],
        columns=['_time', '_value', '_type']
    ).sort_values('_time')
    bpm_df = bpm_df.loc[
        bpm_df['_type'] == 14
        ].filter(items=['_time', '_value'])
    bpm_df['_value'] /= 1000

    if '_BPMChanges' in beatmap:
        bpm_df = bpm_df.append(pd.DataFrame(beatmap['_BPMChanges'], columns=['_time', '_BPM'])
                               .sort_values('_time')
                               .rename(columns={'_BPM': '_value'}),
                               ignore_index=True)

    return bpm_df.loc[bpm_df['_value'] >= 30]  # BPM can't be zero


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


def beatmap2beat_df(beatmap: JSON, info: JSON) -> pd.DataFrame:
    # Load notes
    df = pd.DataFrame(
        (x for x in beatmap['_notes'] if '_time' in x),  # TODO: Remove
        columns=['_time', '_type', '_lineLayer', '_lineIndex', '_cutDirection', ]
    ).sort_values('_time')

    # Throw away bombs
    df = df.loc[df['_type'] != 3]

    df = df.sort_values(by=['_time', '_lineLayer'])
    df = to_categorical(df)

    # Round to 2 decimal places for normalization for block alignment
    df['_time'] = round(df['_time'], 2)

    # Compute actual time in seconds, not beats
    bpm_df = create_bpm_df(beatmap)
    df['_time'] = np.around(compute_true_time(df['_time'].to_numpy(dtype=np.float_),
                                              bpm_df.to_numpy(dtype=np.float_),
                                              info["_beatsPerMinute"]), 3)

    out_df = merge_beat_elements(df)

    out_df = compute_time_cols(out_df)

    out_df.index = out_df.index.rename('time')

    return out_df


def merge_beat_elements(df):
    """
    Per each beat each hand should have exactly one beat element.
    :param df: beat elements
    :return:
    """
    hands = [df.loc[df['_type'] == x]
                 .drop_duplicates('_time', 'last')
                 .set_index('_time')
                 .drop(columns='_type')
             for x, hand in [[0, 'l'], [1, 'r']]]
    for hand in [0, 1]:
        not_in = hands[hand - 1].index.difference(hands[hand].index)
        hands[hand] = hands[hand].append(hands[hand - 1].loc[not_in])
    hands = [x.add_prefix(hand) for x, hand in zip(hands, ['l', 'r'])]
    out_df = pd.concat(hands, axis=1)
    return out_df


def path2beat_df(beatmap_path, info_path) -> pd.DataFrame:
    with open(info_path) as info_data:
        info = json.load(info_data)
        if 'beatsPerMinute' in info:
            info['_beatsPerMinute'] = info['beatsPerMinute']
    with open(beatmap_path) as beatmap_data:
        beatmap = json.load(beatmap_data)
        return beatmap2beat_df(beatmap, info)


def process_song_folder(folder, config: Config, order=(0, 1)):
    progress(*order, config=config, name='Processing song folders')
    print(folder)

    files = []
    for dirpath, dirnames, filenames in os.walk(folder):
        files.extend(filenames)
        break
    info_path = os.path.join(folder, [x for x in files if 'info' in x][0])
    file_ogg = os.path.join(folder, [x for x in files if x.endswith('gg')][0])
    folder_name = folder.split('/')[-1]
    df_difficulties = []

    try:
        mfcc_df = path2mfcc_df(file_ogg, config=config)
    except ValueError as e:
        print(f'\tSkipped file {folder_name}  |  {folder}:\n\t\t{e}', file=stderr)
        return None

    for difficulty in ['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus']:
        beatmap_path = [x for x in files if difficulty in x]
        if beatmap_path:
            try:
                beatmap_path = os.path.join(folder, beatmap_path[0])
                df = path2beat_df(beatmap_path, info_path)
                df = join_closest_index(df, mfcc_df)

                df['difficulty'] = difficulty
                df['name'] = folder_name
                df = df.set_index(['name', 'difficulty'], append=True).reorder_levels(['name', 'difficulty', 'time'])

                df_difficulties.append(df)
            except (IndexError, KeyError, UnicodeDecodeError) as e:
                print(f'\tSkipped file {folder_name}/{difficulty}  |  {folder}:\n\t\t{e}', file=stderr)

    if df_difficulties:
        return pd.concat(df_difficulties)
    return None


def join_closest_index(df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """
    Join `df` with the closest row (by index) of `other`
    :param df: index in time,
    :param other: index in time, constant intervals
    :return: df
    """
    original_index = df.index
    round_index = other.index.values[1] - other.index.values[0]
    df.index = np.floor(df.index / round_index).astype(int)
    other.index = (other.index / round_index).astype(int)

    if not other.name:
        other.name = 'other'
    df = df.join(other)
    df.index = original_index
    return df


def path2mfcc_df(file_ogg, config: Config) -> pd.DataFrame:
    file_cache = f'{".".join(file_ogg.split(".")[:-1])}.pkl'
    if config.audio_processing['use_cache'] and os.path.exists(file_cache):
        return pd.read_pickle(file_cache)

    signal, samplerate = sf.read(file_ogg)
    df = audio2mfcc_df(signal, samplerate, config)
    df.to_pickle(file_cache)
    return df


def get_mfcc(
        signal: np.ndarray,
        sampling_frequency: int,
        frame_length=0.020,
        frame_stride=0.01,
        num_cepstral=13,
        num_filters=40,
        fft_length=512) -> np.ndarray:
    """
    Compatibility wrapper for Sonopy / Soundpy
    :param signal:
    :param sampling_frequency: in Hz
    :param frame_length: in seconds
    :param frame_stride: in seconds
    :param num_cepstral:
    :param num_filters:
    :param fft_length:
    :return: power log + MFCC
    """

    # mfcc = speechpy.feature.mfcc(signal,
    #                              sampling_frequency=samplerate,
    #                              frame_length=config.audio_processing['frame_length'],
    #                              frame_stride=config.audio_processing['frame_stride'],
    #                              num_filters=40,
    #                              fft_length=512,
    #                              low_frequency=0,
    #                              num_cepstral=13)
    # # Normalize
    # mfcc_cmvn = speechpy.processing.cmvnw(mfcc, win_size=301, variance_normalization=True)

    return sonopy.mfcc_spec(signal,
                            sample_rate=sampling_frequency,
                            window_stride=(int(frame_length * 1000),
                                           int(frame_stride * 1000)),
                            fft_size=fft_length,
                            num_filt=num_filters,
                            num_coeffs=num_cepstral,
                            return_parts=False)


def audio2mfcc_df(signal: np.ndarray, samplerate: int, config: Config) -> pd.DataFrame:
    if len(signal) > config.audio_processing['signal_max_length']:
        raise ValueError('[process|audio] Signal longer than set maximum')
    # Stereo to mono
    if signal.shape[1] == 2:
        signal = (signal[:, 0] + signal[:, 1]) / 2
    else:
        signal = signal[:, 0]

    # Pre-emphasize
    # signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)  # TODO: should be used?

    # Extract MFCC features
    mfcc = get_mfcc(signal,
                    sampling_frequency=samplerate,
                    frame_length=config.audio_processing['frame_length'],
                    frame_stride=config.audio_processing['frame_stride'],
                    num_filters=40,
                    fft_length=512,
                    num_cepstral=13)

    # Compute the time index
    index = np.arange(0,
                      (len(mfcc) - 0.5) * config.audio_processing['frame_stride'],
                      config.audio_processing['frame_stride']) + config.audio_processing['frame_length']
    df = pd.DataFrame(data=mfcc, index=index)

    if config.audio_processing['use_temp_derrivatives']:
        df = df.join(df.diff().fillna(0), rsuffix='_d')

    df.name = 'mfcc'
    return df.apply(np.array, axis=1)


if __name__ == '__main__':
    # TODO: Does not work on files with BMP changes
    config = Config()
    config.audio_processing['use_cache'] = False
    df1 = process_song_folder('../data/new_dataformat/3207', config=config)
    df1 = process_song_folder('../data/new_dataformat/3db2', config=config)
    # df1 = path2beat_df('../data/new_dataformat/4b58/ExpertPlus.dat',
    #                    '../data/new_dataformat/4b58/info.dat')
    # df1 = path2df('../data/new_dataformat/5535/ExpertPlus.dat',
    #               '../data/new_dataformat/5535/info.dat')
    # df1 = path2df('../data/new_dataformat/3207/Expert.dat',
    #               '../data/new_dataformat/3207/info.dat')

    print(df1)
