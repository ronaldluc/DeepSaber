from dataclasses import dataclass
from time import time


@dataclass
class Config:
    audio_processing = {
        'frame_length': 0.20,   # in seconds
        'frame_stride': 0.10,   # in seconds
        'use_temp_derrivatives': True,
        'use_cache': True,
        'signal_max_length': 2.5e7,
    }
    utils = {
        'progress_bar_length': 20,
        'progress_bar': True,
    }
    preprocessing = {
        'snippet_window_length': 50,
        'snippet_window_skip': 25,
    }


class Timer:
    def __init__(self):
        self.start = time()

    def __call__(self, name):
        print(f'\r{name:>55}: {time() - self.start}')
        self.start = time()