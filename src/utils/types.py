from dataclasses import dataclass
from time import time


@dataclass
class Config:
    audio_processing = {
        'num_cepstral': 13,
        'frame_length': 0.020,  # in seconds
        'frame_stride': 0.010,  # in seconds
        'time_shift': -0.0,  # in seconds
        'use_temp_derrivatives': True,
        'use_cache': True,
        'signal_max_length': 2.5e7,  # in samples
    }
    utils = {
        'progress_bar_length': 20,
        'progress_bar': True,
    }
    beat_preprocessing = {
        'snippet_window_length': 50,    # number of beats
        'snippet_window_skip': 25,      # number of beats
        'beat_elements': ['l_lineLayer', 'l_lineIndex', 'l_cutDirection',
                          'r_lineLayer', 'r_lineIndex', 'r_cutDirection', ],
    }
    dataset = {
        'beat_elements': beat_preprocessing['beat_elements'],
        'beat_elements_previous_prediction': [f'{x}_prev' for x in beat_preprocessing['beat_elements']],
        'audio': ['mfcc', ],
        'regression': ['prev', 'next', 'part', ]
    }
    training = {
        'data_split': [0.0, 0.8, 0.9, 0.99, ],
    }


class Timer:
    def __init__(self):
        self.start = time()

    def __call__(self, name):
        print(f'\r{name:>55}: {time() - self.start}')
        self.start = time()
