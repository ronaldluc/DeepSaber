from experiments.compute import init_test, eval_config
from process.api import load_datasets, generate_datasets, create_song_list
from utils.functions import dataset_stats

from utils.types import Config, DatasetConfig, ModelType


def main():
    mainly_id()
    mainly_vec()


def mainly_id():
    base_folder, return_list, test, timer, train, val = init_test()
    prefix = 'id_'
    test_name = 'information_comparison'
    print('Running information comparison')

    csv_file = base_folder / 'temp' / f'{prefix}{test_name}.csv'

    dataset = DatasetConfig()
    ALL = [dataset.categorical + dataset.audio + dataset.regression, ]

    # Common configuration
    config = Config()
    config.training.model_type = ModelType.CUSTOM
    config.training.cnn_repetition = 0
    config.training.lstm_repetition = 2
    config.training.dense_repetition = 0
    config.training.model_size = 512
    config.training.dropout = 0.4
    config.training.initial_learning_rate = 1e-2
    config.training.batch_size = 128
    config.training.label_smoothing = 0.5
    config.training.mixup_alpha = 0.5
    config.training.l2_regularization = 0
    hp = None

    config.dataset.beat_maps_folder = config.dataset.beat_maps_folder.parent / 'new_dataformat'
    config.dataset.storage_folder = base_folder / 'new_datasets'
    song_folders = create_song_list(config.dataset.beat_maps_folder)

    # First generate all data using all of the audio features
    config.audio_processing.use_temp_derrivatives = True
    config.audio_processing.time_shift = -0.4
    # generate_datasets(song_folders, config)
    train, val, test = load_datasets(config)
    dataset_stats(train)

    for repetition in range(7):
        config.training.x_groups = ALL
        config.training.y_groups = [['word_id'], ]
        configuration_name = f'X: {config.training.x_groups}\nY: {config.training.y_groups}'

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_vec'], ]
        config.training.y_groups = [['word_vec'], ]
        configuration_name = f'X: {config.training.x_groups}\nY: {config.training.y_groups}'

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_id'], ]
        config.training.y_groups = [['word_id'], ]
        configuration_name = f'X: {config.training.x_groups}\nY: {config.training.y_groups}'

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_id'], ['mfcc'], ]
        config.training.y_groups = [['word_id'], ]
        configuration_name = f"X: {[['prev_word_id'], ['MFCC', 'dMFCC']]}\nY: {config.training.y_groups}"

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_id'], ['prev', 'next'], ]
        config.training.y_groups = [['word_id'], ]
        configuration_name = f"X: {config.training.x_groups}\nY: {config.training.y_groups}"

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_id'], ['part'], ]
        config.training.y_groups = [['word_id'], ]
        configuration_name = f"X: {config.training.x_groups}\nY: {config.training.y_groups}"

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_id'], ['difficulty'], ]
        config.training.y_groups = [['word_id'], ]
        configuration_name = f"X: {config.training.x_groups}\nY: {config.training.y_groups}"

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_id'], ] + ALL
        config.training.y_groups = [['word_id'], ]
        configuration_name = f'X: {config.training.x_groups}\nY: {config.training.y_groups}'

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_vec', 'prev_word_id'], ] + ALL
        config.training.y_groups = [['word_id'], ]
        configuration_name = f'X: {config.training.x_groups}\nY: {config.training.y_groups}'

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_vec', 'prev_word_id'], ] + ALL
        config.training.y_groups = [['word_vec', 'word_id'], ]
        configuration_name = f'X: {config.training.x_groups}\nY: {config.training.y_groups}'

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)
        pass

    # Generate without derivatives
    config.dataset.beat_maps_folder = config.dataset.beat_maps_folder.parent / 'test_new_dataformat'
    config.dataset.storage_folder = base_folder / 'test_new_datasets'
    config.audio_processing.use_temp_derrivatives = False
    config.audio_processing.time_shift = None
    generate_datasets(song_folders, config)
    train, val, test = load_datasets(config)
    for repetition in range(7):
        config.training.x_groups = [['prev_word_id'], ['mfcc', ]]
        config.training.y_groups = [['word_id'], ]
        configuration_name = f'X: {config.training.x_groups}\nY: {config.training.y_groups}'

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)
        pass


def mainly_vec():
    base_folder, return_list, test, timer, train, val = init_test()
    prefix = 'vec'
    test_name = 'information_comparison'
    print('Running information comparison')

    csv_file = base_folder / 'temp' / f'{prefix}{test_name}.csv'

    dataset = DatasetConfig()
    ALL = [dataset.categorical + dataset.audio + dataset.regression, ]

    # Common configuration
    config = Config()
    config.training.model_type = ModelType.CUSTOM
    config.training.cnn_repetition = 0
    config.training.lstm_repetition = 2
    config.training.dense_repetition = 0
    config.training.model_size = 512
    config.training.dropout = 0.4
    config.training.initial_learning_rate = 1e-2
    config.training.batch_size = 128
    config.training.label_smoothing = 0.5
    config.training.mixup_alpha = 0.5
    config.training.l2_regularization = 0
    hp = None

    config.dataset.beat_maps_folder = config.dataset.beat_maps_folder.parent / 'test_new_dataformat'
    config.dataset.storage_folder = base_folder / 'test_new_datasets'
    song_folders = create_song_list(config.dataset.beat_maps_folder)

    # First generate all data using all of the audio features
    config.audio_processing.use_temp_derrivatives = True
    config.audio_processing.time_shift = -0.4
    dataset_stats(train)

    for repetition in range(7):
        config.training.x_groups = ALL
        config.training.y_groups = [['word_vec'], ]
        configuration_name = f'X: {config.training.x_groups}\nY: {config.training.y_groups}'

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_vec'], ]
        config.training.y_groups = [['word_id'], ]
        configuration_name = f'X: {config.training.x_groups}\nY: {config.training.y_groups}'

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_vec'], ['mfcc'], ]
        config.training.y_groups = [['word_id'], ]
        configuration_name = f"X: {[['prev_word_id'], ['MFCC', 'dMFCC']]}\nY: {config.training.y_groups}"

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_vec'], ['prev', 'next'], ]
        config.training.y_groups = [['word_id'], ]
        configuration_name = f"X: {config.training.x_groups}\nY: {config.training.y_groups}"

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_vec'], ['part'], ]
        config.training.y_groups = [['word_id'], ]
        configuration_name = f"X: {config.training.x_groups}\nY: {config.training.y_groups}"

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_vec'], ] + ALL
        config.training.y_groups = [['word_id'], ]
        configuration_name = f'X: {config.training.x_groups}\nY: {config.training.y_groups}'

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_vec', 'prev_word_id'], ] + ALL
        config.training.y_groups = [['word_id'], ]
        configuration_name = f'X: {config.training.x_groups}\nY: {config.training.y_groups}'

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config.training.x_groups = [['prev_word_vec', 'prev_word_id'], ] + ALL
        config.training.y_groups = [['word_vec', 'word_id'], ]
        configuration_name = f'X: {config.training.x_groups}\nY: {config.training.y_groups}'

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)
        pass


if __name__ == '__main__':
    main()
