from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from utils.types import Config


def progress(done: int, total: int, config: Config, name=''):
    one_part = max(1, total // config.utils.progress_bar_length)

    if config.utils.progress_bar and done % one_part == 0:
        print(f'\r{name:>55} | {"#" * (done // one_part)}{"-" * ((total - done) // one_part)} | {total:>7}',
              flush=True,
              end='')


def check_consistency(df: pd.DataFrame):
    for col in df.columns:
        num = np.array(df[col].to_list())
        if len(num.shape) <= 1 and isinstance(df[col].iloc[0], (np.ndarray, list)):
            raise ValueError(f'[check consistency] failed on {col} with shape {num.shape}'
                             f' and first row of type {type(df[col].iloc[0])}')

    return True


def y2action_word(y: Dict[str, tf.TensorArray]):
    """
    Converts dictionary of action one-hot vectors into a action word representation
    Example output element: L000_R001
    """
    word = []

    for hand in 'lr':
        word += [hand.upper()]
        word += [tf.strings.as_string(tf.argmax(y[f'{hand}_{name}'], axis=-1)) for name in
                 ['lineLayer', 'lineIndex', 'cutDirection']]
        word += ['_']

    return tf.strings.join(word[:-1])


def create_word_mapping(action_model):
    word_id = {key: val + 2 for key, val in zip(action_model.vocab.keys(), range(len(action_model.vocab)))}
    word_id['MASK'] = 0
    word_id['UNK'] = 1
    return word_id