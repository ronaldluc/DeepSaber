from dataclasses import dataclass, field
from time import time
from typing import Tuple, Dict, Callable
from typing import Union, Mapping, List

import gensim

JSON = Union[str, int, float, bool, None, Mapping[str, 'JSON'], List['JSON']]
from pathlib import Path

ROOT_DIR: Path = Path(__file__).parents[2]


@dataclass
class AudioProcessingConfig:
    num_cepstral: int = 13
    frame_length: float = 0.010  # in seconds
    frame_stride: float = 0.010  # in seconds
    time_shift: float = -0.0  # in seconds
    use_temp_derrivatives: float = True
    use_cache: bool = True
    signal_max_length: float = 2.5e7  # in samples


@dataclass
class UtilsConfig:
    progress_bar_length: int = 20
    progress_bar: bool = True


@dataclass
class BeatPreprocessingConfig:
    snippet_window_length: bool = 50  # in the number of beats
    snippet_window_skip: bool = 25  # in the number of beats
    beat_elements: List = field(
        default_factory=lambda: ['l_lineLayer', 'l_lineIndex', 'l_cutDirection',
                                 'r_lineLayer', 'r_lineIndex', 'r_cutDirection', ])
    beat_actions: List = field(
        default_factory=lambda: ['word_vec', 'word_id'])


@dataclass
class DatasetConfig:
    beat_maps_folder: Path = ROOT_DIR / 'data/human_beatmaps/new_dataformat'
    storage_folder: Path = ROOT_DIR / 'data/new_datasets'
    action_word_model_path: Path = storage_folder / 'fasttext.model'  # gensim FastText.KeyedVectors class
    # num_classes: Dict = field(
    #     default_factory=lambda: {'difficulty': 5,  # ending of the column name: number of classes
    #                              '_lineLayer': 3, '_lineIndex': 4, '_cutDirection': 9,
    #                              'word_id': len(gensim.models.KeyedVectors.load(
    #                                  str(DatasetConfig.action_word_model_path)).vocab) + 2})
    difficulty_mapping: Dict = field(
        default_factory=lambda: {d: enum for enum, d in enumerate(['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus'])})

    # dataset groups
    beat_elements: List = field(
        default_factory=lambda: BeatPreprocessingConfig().beat_elements)
    beat_actions: List = field(
        default_factory=lambda: BeatPreprocessingConfig().beat_actions)
    beat_elements_previous_prediction: List = field(
        default_factory=lambda: [f'prev_{x}' for x in BeatPreprocessingConfig().beat_elements])
    beat_actions_previous_prediction: List = field(
        default_factory=lambda: [f'prev_{x}' for x in BeatPreprocessingConfig().beat_actions])
    categorical: List = field(
        default_factory=lambda: ['difficulty', ])
    audio: List = field(
        default_factory=lambda: ['mfcc', ])
    regression: List = field(
        default_factory=lambda: ['prev', 'next', 'part', ])
    _word_id_num_classes: int = 0

    @property
    def num_classes(self):
        return {'difficulty': 5,  # ending of the column name: number of classes
                '_lineLayer': 3, '_lineIndex': 4, '_cutDirection': 9,
                'word_id': self.word_id_num_classes}

    @property
    def word_id_num_classes(self):
        if self._word_id_num_classes > 0:
            return self._word_id_num_classes
        if self.action_word_model_path.exists():
            self._word_id_num_classes = len(gensim.models.KeyedVectors.load(str(self.action_word_model_path)).vocab) + 2
        return self._word_id_num_classes


@dataclass
class TrainingConfig:
    cnn_repetition: int = 2
    lstm_repetition: int = 2
    dense_repetition: int = 2
    model_size: int = 256 + 128
    dropout: float = 0.3
    initial_learning_rate: float = 8e-3
    data_split: Tuple = (0.0, 0.8, 0.9, 0.99,)
    AVS_proxy_ratio: float = 0.2  # Fraction of songs to compute AVS cosine similarity
    # if word reconstruction has to be used
    batch_size: float = 128
    label_smoothing: float = 0.5
    mixup_alpha: float = 0.5  # `mixup_alpha` == 0 => mixup is not used
    use_difficulties: List = field(
        default_factory=lambda: ['Normal', 'Hard', 'Expert', ])
    categorical_groups: List = field(
        default_factory=lambda: [DatasetConfig().beat_elements, DatasetConfig().beat_elements_previous_prediction,
                                 DatasetConfig().categorical, ['word_id', 'prev_word_id', ]])

    # in dataset groups List(List(str))
    regression_groups: Tuple = field(
        default_factory=lambda: [DatasetConfig().audio, DatasetConfig().regression,
                                 ['word_vec', 'prev_word_vec', ]])  # in dataset groups
    x_groups: Tuple = field(
        default_factory=lambda: [
            # DatasetConfig().beat_elements_previous_prediction,
            # ['prev_word_id', ],
            ['prev_word_vec', ],
            DatasetConfig().categorical, DatasetConfig().audio, DatasetConfig().regression
        ])
    y_groups: List = field(
        default_factory=lambda: [
            # DatasetConfig().beat_elements,
            # DatasetConfig().beat_actions,
            # ['word_vec', ],
            ['word_id', ],
        ])


def temperature(steps):
    return 0.7 + 1 / (steps + 3)


@dataclass
class GenerationConfig:
    # temperature = 0.7
    restrict_vocab: int = 500  # use only the first # actions. `None` == use all
    temperature: Callable = temperature


@dataclass
class Config:
    audio_processing: AudioProcessingConfig = field(default_factory=AudioProcessingConfig)
    utils: UtilsConfig = field(default_factory=UtilsConfig)
    beat_preprocessing: BeatPreprocessingConfig = field(default_factory=BeatPreprocessingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    base_data_folder: Path = ROOT_DIR / 'data'
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
