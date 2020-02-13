import numba

from utils.types import Config


def progress(done: int, total: int, config: Config, name=''):
    one_part = max(1, total // config.utils['progress_bar_length'])

    if config.utils['progress_bar'] and done % one_part == 0:
        print(f'\r{name:>55} | {"#" * (done // one_part)}{"-" * ((total - done) // one_part)} | {total:6}',
              flush=True,
              end='')
