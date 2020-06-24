from dataclasses import dataclass, field
from time import time
from typing import Union, Mapping, List

import gensim

JSON = Union[str, int, float, bool, None, Mapping[str, 'JSON'], List['JSON']]
from pathlib import Path

ROOT_DIR: Path = Path(__file__).parents[2]

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
    beat_actions = ['word_vec', 'word_id']


@dataclass
class DatasetConfig:
    beat_maps_folder = ROOT_DIR / 'data/human_beatmaps'
    storage_folder = ROOT_DIR / 'data/new_datasets'
    action_word_model_path = storage_folder / 'fasttext.model'  # gensim FastText.KeyedVectors class
    num_classes = {'difficulty': 5,  # ending of the column name: number of classes
                   '_lineLayer': 3, '_lineIndex': 4, '_cutDirection': 9,
                   'word_id': len(gensim.models.KeyedVectors.load(str(action_word_model_path)).vocab) + 2}
    difficulty_mapping = {d: enum for enum, d in enumerate(['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus'])}

    # dataset groups
    beat_elements = BeatPreprocessingConfig.beat_elements
    beat_actions = BeatPreprocessingConfig.beat_actions
    beat_elements_previous_prediction = [f'prev_{x}' for x in BeatPreprocessingConfig.beat_elements]
    beat_actions_previous_prediction = [f'prev_{x}' for x in BeatPreprocessingConfig.beat_actions]
    categorical = ['difficulty', ]
    audio = ['mfcc', ]
    regression = ['prev', 'next', 'part', ]


@dataclass
class TrainingConfig:
    data_split = (0.0, 0.8, 0.9, 0.99,)
    AVS_proxy_ratio = 0.1  # Fraction of songs to compute AVS cosine similarity if word reconstruction has to be used
    batch_size = 256
    label_smoothing = 0.0
    mixup_alpha = 0.0  # `mixup_alpha` == 0 => mixup is not used
    use_difficulties = ['Normal', 'Hard', 'Expert']
    categorical_groups = [DatasetConfig.beat_elements, DatasetConfig.beat_elements_previous_prediction,
                          DatasetConfig.categorical, ['word_id', 'prev_word_id']]
    # in dataset groups List[List[str]]
    regression_groups = [DatasetConfig.audio, DatasetConfig.regression,
                         ['word_vec', 'prev_word_vec']]  # in dataset groups
    x_groups = [
        # DatasetConfig.beat_elements_previous_prediction,
        # ['prev_word_id'],
        ['prev_word_vec'],
        DatasetConfig.categorical, DatasetConfig.audio, DatasetConfig.regression]
    # x_groups = [['prev_word_vec'], DatasetConfig.beat_actions_previous_prediction, DatasetConfig.categorical,
    #             DatasetConfig.audio, DatasetConfig.regression]
    # y_groups = [DatasetConfig.beat_elements]
    # y_groups = [DatasetConfig.beat_actions]
    y_groups = [['word_id'], ]
    # y_groups = [['word_vec'], ]


@dataclass
class GenerationConfig:
    temperature = 0.7
    restrict_vocab = 500  # use only the first # actions. `None` == use all


@dataclass
class Config:
    audio_processing: AudioProcessingConfig = field(default_factory=AudioProcessingConfig)
    utils: UtilsConfig = field(default_factory=UtilsConfig)
    beat_preprocessing: BeatPreprocessingConfig = field(default_factory=BeatPreprocessingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    base_data_folder = ROOT_DIR / 'data'
    # audio_processing = AudioProcessingConfig()
    # utils = UtilsConfig()
    # beat_preprocessing = BeatPreprocessingConfig()
    # dataset = DatasetConfig()
    # training = TrainingConfig()
    # generation = GenerationConfig()
    # base_data_folder = ROOT_DIR / 'data'


class Timer:
    def __init__(self):
        self.start = time()

    def __call__(self, name, level=5):
        diff = time() - self.start
        print(f'\r{name:>{24 + level * 12}}: {diff}')
        self.start = time()
        return diff
