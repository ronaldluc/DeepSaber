import json
import os
import signal
from multiprocessing.pool import Pool
from sys import stderr

import numba
import numpy as np
import pandas as pd
import soundfile as sf
import speechpy

from utils.functions import progress
from utils.types import Config, JSON


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
    df['prev'] = df['time'].diff().astype('float32')
    df['prev'] = df['prev'].fillna(df['prev'].max())
    # next beat in seconds
    df['next'] = df['prev'].shift(periods=-1).astype('float32')
    df['next'] = df['next'].fillna(df['next'].max())
    # which part of the song each beat belongs to
    df['part'] = (df['time'] / df['time'].max()).astype('float32')
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


def beatmap2beat_df(beatmap: JSON, info: JSON, config: Config) -> pd.DataFrame:
    # Load notes
    df: pd.DataFrame = pd.DataFrame(
        (x for x in beatmap['_notes'] if '_time' in x),
        columns=['_time', '_type', '_lineLayer', '_lineIndex', '_cutDirection', ],
    ).astype(dtype={'_type': 'int8', '_lineLayer': 'int8', '_lineIndex': 'int8', '_cutDirection': 'int8'}, ) \
        .sort_values('_time')

    # Throw away bombs
    df = df.loc[df['_type'] != 3]

    df = df.sort_values(by=['_time', '_lineLayer'])

    # Round to 2 decimal places for normalization for block alignment
    df['_time'] = round(df['_time'], 2)

    # Compute actual time in seconds, not beats
    bpm_df = create_bpm_df(beatmap)
    df['_time'] = np.around(compute_true_time(df['_time'].to_numpy(dtype=np.float_),
                                              bpm_df.to_numpy(dtype=np.float_),
                                              info["_beatsPerMinute"]), 3)

    out_df = merge_beat_elements(df)

    out_df['word'] = compute_action_words(df)

    check_column_ranges(out_df, config)

    out_df = compute_time_cols(out_df)

    out_df.index = out_df.index.rename('time')

    return out_df


def compute_action_words(df):
    """
    Transform all beat elements with the same time stamp into one action, represented by a word.
    Example: [{hand: L, _lineLayer: 0, _lineIndex: 1, _cutDirection: 2},
              {hand: R, _lineLayer: 2, _lineIndex: 3, _cutDirection: 8}] -> 'L012_R238'
    """
    df = df.set_index('_time')
    df['hand'] = 'L'
    df.loc[df['_type'] == 1, 'hand'] = 'R'
    df['word'] = df['hand'].str.cat([df[x].astype(str) for x in ['_lineLayer', '_lineIndex', '_cutDirection']])
    df = df.sort_values('word')
    temp = df['word'].groupby(level=0).apply(lambda x: x.str.cat(sep='_'))
    return temp


def check_column_ranges(out_df, config):
    for col in config.beat_preprocessing.beat_elements:
        minimum, maximum = out_df[col].min(), out_df[col].max()
        num_classes = [num for ending, num in config.dataset.num_classes.items() if col.endswith(ending)][0] - 1
        if minimum < 0 or num_classes < maximum:
            raise ValueError(
                f'[process|compute] column {col} with range <{minimum}, {maximum}> outside range <0, {num_classes}>')


def merge_beat_elements(df: pd.DataFrame):
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


def path2beat_df(beatmap_path, info_path, config: Config) -> pd.DataFrame:
    with open(info_path) as info_data:  # normalize across old and new version of beatmap files
        info = json.load(info_data)
        if 'beatsPerMinute' in info:
            info['_beatsPerMinute'] = info['beatsPerMinute']
    with open(beatmap_path) as beatmap_data:
        beatmap = json.load(beatmap_data)
        return beatmap2beat_df(beatmap, info, config)


def process_song_folder(folder, config: Config, order=(0, 1)):
    """
    Return processed and concatenated dataframe of all songs in `folder`.
    Returns `None` if an error occurs.

    Each beat is determined by multiindex of song name, difficulty and time (in seconds).
    Each beat contains information about:
    - MFCC of audio
    - beat elements
    - previous beat elements
    - time (in seconds) to previous / next beat
    - proportion of the song it belongs to
    """
    progress(*order, config=config, name='Processing song folders')

    files = []  # TODO: Rewrite to use Pathlib
    for dirpath, dirnames, filenames in os.walk(folder):
        files.extend(filenames)
        break
    info_path = os.path.join(folder, [x for x in files if 'info' in x][0])
    file_ogg = os.path.join(folder, [x for x in files if x.endswith('gg')][0])
    folder_name = folder.split('/')[-1]
    df_difficulties = []

    try:
        mfcc_df = path2mfcc_df(file_ogg, config=config)
    except (ValueError, FileNotFoundError, AttributeError) as e:  # TODO: Remove AttributeError if not necessary
        print(f'\n\t[process | process_song_folder] Skipped file {folder_name}  |  {folder}:\n\t\t{e}', file=stderr)
        return None

    for difficulty in ['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus']:
        beatmap_path = [x for x in files if difficulty in x]
        if beatmap_path:
            try:
                beatmap_path = os.path.join(folder, beatmap_path[0])
                df = path2beat_df(beatmap_path, info_path, config)
                df = join_closest_index(df, mfcc_df, 'mfcc')
                df = add_multiindex(df, difficulty, folder_name)

                df_difficulties.append(df)
            except (ValueError, IndexError, KeyError, UnicodeDecodeError) as e:
                print(
                    f'\n\t[process | process_song_folder] Skipped file {folder_name}/{difficulty}  |  {folder}:\n\t\t{e}',
                    file=stderr)

    if df_difficulties:
        return pd.concat(df_difficulties)
    return None


def add_multiindex(df, difficulty, folder_name):
    df['difficulty'] = difficulty
    df['name'] = folder_name
    df = df.set_index(['name', 'difficulty'], append=True).reorder_levels(['name', 'difficulty', 'time'])
    return df


def add_previous_prediction(df: pd.DataFrame, config: Config):
    beat_elements_pp = config.dataset.beat_elements_previous_prediction
    beat_actions_pp = config.dataset.beat_actions_previous_prediction
    df_shifted = df[config.dataset.beat_elements + config.dataset.beat_actions].shift(1)
    df[beat_elements_pp + beat_actions_pp] = df_shifted
    df = df.dropna().copy()
    df.loc[:, beat_elements_pp] = df[beat_elements_pp].astype('int8')

    # Name and difficulty information is contained in the grouping operation
    indexes_to_drop = ['name', 'difficulty']
    df = df.reset_index(level=indexes_to_drop).drop(columns=indexes_to_drop)
    return df


def join_closest_index(df: pd.DataFrame, other: pd.DataFrame, other_name: str = 'other') -> pd.DataFrame:
    """
    Join `df` with the closest row (by index) of `other`
    :param df: index in time,
    :param other: index in time, constant intervals
    :param other_name: name of the joined columns
    :return: df
    """
    original_index = df.index
    round_index = other.index.values[1] - other.index.values[0]
    df.index = np.floor(df.index / round_index).astype(int)
    other.index = (other.index / round_index).astype(int)
    other = other.reset_index(drop=True)

    other.name = other_name
    df = df.join(other)
    df.index = original_index
    return df


def path2mfcc_df(ogg_path, config: Config) -> pd.DataFrame:
    cache_path = f'{".".join(ogg_path.split(".")[:-1])}.pkl'

    if os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)
    else:
        if config.audio_processing.use_cache:
            raise FileNotFoundError('Cache file not found')
        signal, samplerate = sf.read(ogg_path)
        df = audio2mfcc_df(signal, samplerate, config)
        df.to_pickle(cache_path)

    if config.audio_processing.use_temp_derrivatives:
        df = df.join(df.diff().fillna(0), rsuffix='_d')

    df.index = df.index + config.audio_processing.time_shift

    flatten = np.split(df.to_numpy().astype('float16').flatten(), len(df.index))
    return pd.DataFrame(data={'mfcc': flatten},
                        index=df.index)


def audio2mfcc_df(signal: np.ndarray, samplerate: int, config: Config) -> pd.DataFrame:
    if len(signal) > config.audio_processing.signal_max_length:
        raise ValueError('[process|audio] Signal longer than set maximum')

    # Stereo to mono
    if signal.shape[1] == 2:
        signal = (signal[:, 0] + signal[:, 1]) / 2
    else:
        signal = signal[:, 0]

    # Pre-emphasize
    # signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)  # TODO: should be used?

    # Extract MFCC features
    mfcc = speechpy.feature.mfcc(signal,
                                 sampling_frequency=samplerate,
                                 frame_length=config.audio_processing.frame_length,
                                 frame_stride=config.audio_processing.frame_stride,
                                 num_filters=40,
                                 fft_length=512,
                                 num_cepstral=config.audio_processing.num_cepstral)

    # Compute the time index
    index = np.arange(0,
                      (len(mfcc) - 0.5) * config.audio_processing.frame_stride,
                      config.audio_processing.frame_stride) + config.audio_processing.frame_length
    return pd.DataFrame(data=mfcc, index=index, dtype='float16')


def init_worker():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def create_ogg_cache(ogg_path, config: Config, order=(0, 1)):
    progress(*order, config=config, name='Recalculating MFCCs')
    try:
        path2mfcc_df(ogg_path, config=config)
    except ValueError as e:
        print(f'\tSkipped file {ogg_path} \n\t\t{e}', file=stderr)


def create_ogg_caches(ogg_paths, config: Config):
    total = len(ogg_paths)
    inputs = ((s, config, (i, total)) for i, s in enumerate(ogg_paths))
    pool = Pool(initializer=init_worker())
    pool.starmap(create_ogg_cache, inputs)
    pool.close()
    pool.join()


def remove_ogg_cache(ogg_paths):
    for i, ogg_path in enumerate(ogg_paths):
        cache_path = f'{".".join(ogg_path.split(".")[:-1])}.pkl'
        if os.path.exists(cache_path):
            os.remove(cache_path)


def create_ogg_paths(song_folders):
    ogg_paths = []
    for folder in song_folders:
        files = []
        for dirpath, dirnames, filenames in os.walk(folder):
            files.extend(filenames)
            break
        ogg_paths.append(os.path.join(folder, [x for x in files if x.endswith('gg')][0]))
    return ogg_paths


if __name__ == '__main__':
    # TODO: Does not work on files with BMP changes
    config = Config()
    # config.audio_processing.use_cache = False
    df1 = process_song_folder('../data/new_dataformat/3207', config=config)
    print(df1.columns)

    df1 = process_song_folder('../data/new_dataformat/3db2', config=config)
    # df1 = path2beat_df('../data/new_dataformat/4b58/ExpertPlus.dat',
    #                    '../data/new_dataformat/4b58/info.dat')
    # df1 = path2df('../data/new_dataformat/5535/ExpertPlus.dat',
    #               '../data/new_dataformat/5535/info.dat')
    # df1 = path2df('../data/new_dataformat/3207/Expert.dat',
    #               '../data/new_dataformat/3207/info.dat')

    print(df1)


def generate_snippets(song_df: pd.DataFrame, config: Config):
    stack = []
    ln = len(song_df)
    window = config.beat_preprocessing.snippet_window_length
    skip = config.beat_preprocessing.snippet_window_skip

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
