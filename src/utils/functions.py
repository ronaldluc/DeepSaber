from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

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


def dataset_stats(df: pd.DataFrame):
    print(df)
    group_over = ['name', 'difficulty', 'snippet', 'time', ]
    for end_index in range(1, len(group_over) + 1):
        print(f"{df.groupby(group_over[:end_index]).ngroups:9} {' × '.join(group_over[:end_index])}")


def list2numpy(batch, col_name, groupby=('name')):
    return np.array(batch.groupby(list(groupby))[col_name].apply(list).to_list())


def debug_model(model: keras.Model):
    for layer in model.layers:
        shapes = [x.shape for x in layer.weights]
        print(f'{layer.name:12}: {shapes}')
    model.summary()


def name_generator(prefix):
    id_ = 0
    while True:
        yield f'{prefix}{id_}'
        id_ += 1
