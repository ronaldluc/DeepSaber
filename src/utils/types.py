from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Union, Mapping, List

JSON = Union[str, int, float, bool, None, Mapping[str, 'JSON'], List['JSON']]


@dataclass
class AudioProcessingConfig:
    num_cepstral = 13
    frame_length = 0.010  # in seconds
    frame_stride = 0.010  # in seconds
    time_shift = -0.0  # in seconds
    use_temp_derrivatives = True
    use_cache = True
    signal_max_length = 2.5e7  # in samples


@dataclass
class UtilsConfig:
    progress_bar_length = 20
    progress_bar = True


@dataclass
class BeatPreprocessingConfig:
    snippet_window_length = 50  # in the number of beats
    snippet_window_skip = 25  # in the number of beats
    beat_elements = ['l_lineLayer', 'l_lineIndex', 'l_cutDirection',
                     'r_lineLayer', 'r_lineIndex', 'r_cutDirection', ]


@dataclass
class DatasetConfig:
    storage_folder = Path('../data/full_datasets')
    num_classes = {'difficulty': 5,  # ending of the column name: number of classes
                   '_lineLayer': 3, '_lineIndex': 4, '_cutDirection': 9}
    difficulty_mapping = {d: enum for enum, d in enumerate(['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus'])}

    # dataset groups
    beat_elements = BeatPreprocessingConfig.beat_elements
    beat_elements_previous_prediction = [f'prev_{x}' for x in BeatPreprocessingConfig.beat_elements]
    categorical = ['difficulty', ]
    audio = ['mfcc', ]
    regression = ['prev', 'next', 'part', ]


@dataclass
class TrainingConfig:
    data_split = (0.0, 0.8, 0.9, 0.99,)
    batch_size = 512
    label_smoothing = 0.0
    use_difficulties = ['Normal', 'Hard', 'Expert']
    categorical_groups = [DatasetConfig.beat_elements, DatasetConfig.beat_elements_previous_prediction,
                          DatasetConfig.categorical]
    # in dataset groups
    regression_groups = [DatasetConfig.audio, DatasetConfig.regression]  # in dataset groups
    x_groups = [DatasetConfig.beat_elements_previous_prediction, DatasetConfig.categorical,
                DatasetConfig.audio, DatasetConfig.regression]
    y_groups = [DatasetConfig.beat_elements]


@dataclass
class Config:
    audio_processing = AudioProcessingConfig()
    utils = UtilsConfig()
    beat_preprocessing = BeatPreprocessingConfig()
    dataset = DatasetConfig()
    training = TrainingConfig()


class Timer:
    def __init__(self):
        self.start = time()

    def __call__(self, name, level=5):
        print(f'\r{name:>{24 + level * 12}}: {time() - self.start}')
        self.start = time()
