from dataclasses import dataclass
from pathlib import Path
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
        'storage_folder': Path('../data/full_datasets'),
        'num_classes':  {'difficulty': 5,   # ending of the column name: number of classes
                         '_lineLayer': 3, '_lineIndex': 4, '_cutDirection': 9},
        'difficulty_mapping': {d: enum for enum, d in enumerate(['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus'])},

        # dataset groups
        'beat_elements': beat_preprocessing['beat_elements'],
        'beat_elements_previous_prediction': [f'prev_{x}' for x in beat_preprocessing['beat_elements']],
        'categorical': ['difficulty'],
        'audio': ['mfcc', ],
        'regression': ['prev', 'next', 'part', ],
    }
    training = {
        'data_split': (0.0, 0.8, 0.9, 0.99, ),
        'batch_size': 32,
        'use_difficulties': ['Normal', 'Hard', 'Expert'],
        'categorical_groups': ['beat_elements', 'beat_elements_previous_prediction', 'categorical'],   # in dataset groups
        'regression_groups': ['audio', 'regression'],      # in dataset groups
        'x_groups': ['beat_elements_previous_prediction', 'categorical', 'audio', 'regression'],
        'y_groups': ['beat_elements'],
    }


class Timer:
    def __init__(self):
        self.start = time()

    def __call__(self, name, level=5):
        print(f'\r{name:>{24 + level * 12}}: {time() - self.start}')
        self.start = time()
