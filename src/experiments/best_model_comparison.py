import kerastuner as kt

from experiments.compute import init_test, eval_config
from utils.types import Config, DatasetConfig, ModelType


def main():
    base_folder, return_list, test, timer, train, val = init_test()
    prefix = ''
    test_name = 'best_model_comparison2'
    print('Running best model comparison')

    csv_file = base_folder / 'temp' / f'{prefix}{test_name}.csv'

    for repetition in range(7):
        config = Config()
        configuration_name = ModelType.BASELINE
        config.training.model_type = 'baseline'
        config.training.cnn_repetition = 0
        config.training.lstm_repetition = 1
        config.training.dense_repetition = 0
        config.training.model_size = 384
        config.training.dropout = 0
        config.training.initial_learning_rate = 0.001
        config.training.batch_size = 128
        config.training.label_smoothing = 0
        config.training.mixup_alpha = 0
        config.training.l2_regularization = 0
        config.training.x_groups = [['prev_word_id'], ]
        config.training.y_groups = [['word_id'], ]
        hp = None

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config = Config()
        configuration_name = ModelType.DDC
        config.training.model_type = 'ddc'
        config.training.cnn_repetition = 0
        config.training.lstm_repetition = 2
        config.training.dense_repetition = 0
        config.training.model_size = 512
        config.training.dropout = 0.5
        config.training.initial_learning_rate = 0.001
        config.training.batch_size = 64
        config.training.label_smoothing = 0
        config.training.mixup_alpha = 0
        config.training.l2_regularization = 0
        config.training.x_groups = [['prev_word_id'], ['prev', 'next', ]]
        config.training.y_groups = [['word_id'], ]
        hp = None

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config = Config()
        configuration_name = 'Custom vec+id:id'
        config.training.model_type = ModelType.CUSTOM
        config.training.cnn_repetition = 2
        config.training.lstm_repetition = 2
        config.training.dense_repetition = 2
        config.training.model_size = 512
        config.training.dropout = 0.4
        config.training.initial_learning_rate = 1e-2
        config.training.batch_size = 128
        config.training.label_smoothing = 0.5
        config.training.mixup_alpha = 0.5
        config.training.l2_regularization = 0
        config.training.x_groups = [['prev_word_id', 'prev_word_vec'], DatasetConfig().categorical,
                                    DatasetConfig().audio, DatasetConfig().regression]
        config.training.y_groups = [['word_id'], ]
        hp = None

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config = Config()
        configuration_name = 'Custom vec+id:vec'
        config.training.model_type = ModelType.CUSTOM
        config.training.cnn_repetition = 2
        config.training.lstm_repetition = 2
        config.training.dense_repetition = 0
        config.training.model_size = 512
        config.training.dropout = 0.4
        config.training.initial_learning_rate = 1e-2
        config.training.batch_size = 128
        config.training.label_smoothing = 0.5
        config.training.mixup_alpha = 0.5
        config.training.l2_regularization = 0
        config.training.x_groups = [['prev_word_id', 'prev_word_vec'], DatasetConfig().categorical,
                                    DatasetConfig().audio, DatasetConfig().regression]
        config.training.y_groups = [['word_vec'], ]
        hp = None

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        config = Config()
        configuration_name = 'MLSTM'
        config.training.model_type = ModelType.TUNE_MLSTM
        config.training.batch_size = 128
        config.training.label_smoothing = 0.5
        config.training.mixup_alpha = 0.5
        config.training.x_groups = [['prev_word_id', 'prev_word_vec'], DatasetConfig().categorical,
                                    DatasetConfig().audio, DatasetConfig().regression]
        config.training.y_groups = [['word_id'], ]
        hp = kt.HyperParameters()
        fixed_params = {'connections_0': 2,
                        'connections_1': 2,
                        'connections_2': 2,
                        'connections_3': 3,
                        'connections_4': 1,
                        'connections_5': 3,
                        'connections_6': 2,
                        'depth_0': 18,
                        'depth_1': 23,
                        'depth_2': 43,
                        'depth_3': 13,
                        'depth_4': 52,
                        'depth_5': 5,
                        'depth_6': 11,
                        'dropout_0': 0.25612932926324405,
                        'dropout_1': 0.1620424523625309,
                        'dropout_2': 0.4720468723284278,
                        'dropout_3': 0.43881829788147036,
                        'dropout_4': 0.44741780640383355,
                        'dropout_5': 0.3327191857714107,
                        'dropout_6': 0.1367707920005909,
                        'initial_learning_rate': 0.008,
                        'label_smoothing': 0.13716631669361445,
                        'lstm_layers': 3,
                        'width_0': 16,
                        'width_1': 9,
                        'width_2': 15,
                        'width_3': 16,
                        'width_4': 5,
                        'width_5': 11,
                        'width_6': 4,
                        }
        for param, value in fixed_params.items():
            hp.Fixed(param, value=value)

        eval_config(csv_file, timer, return_list, train, val, test, config, test_name, configuration_name, hp)

        pass


if __name__ == '__main__':
    main()
