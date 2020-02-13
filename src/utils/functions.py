import numba
import numpy as np
import pandas as pd

from utils.types import Config


def progress(done: int, total: int, config: Config, name=''):
    one_part = max(1, total // config.utils['progress_bar_length'])

    if config.utils['progress_bar'] and done % one_part == 0:
        print(f'\r{name:>55} | {"#" * (done // one_part)}{"-" * ((total - done) // one_part)} | {total:>7}',
              flush=True,
              end='')


def check_consistency(df: pd.DataFrame):
    for col in df.columns:
        num = np.array(df[col].to_list())
        if len(num.shape) <= 1 and isinstance(df[col].iloc[0], (np.ndarray, list)):
            return False

    return True
