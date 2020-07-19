""" Investigate the influence multiple hyperparameter values for Custom model """

from experiments.compute import eval_hyperparams, init_test

from utils.types import Config, DatasetConfig


def main():
    base_folder, return_list, test, timer, train, val = init_test()
    prefix = 'custom_'

    for repetition in range(7):
        hyper_params = {'mixup_alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25]}
        config = Config()
        config.training.label_smoothing = 0
        eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config, prefix)

        hyper_params = {'label_smoothing': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
        config = Config()
        config.training.mixup_alpha = 0
        eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config, prefix)

        hyper_params = {'label_smoothing': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, ],
                        'mixup_alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, ]}
        config = Config()
        eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config, prefix)

        hyper_params = {'batch_size': [1024, 512, 256, 128, 64, 32]}
        config = Config()
        config.training.mixup_alpha = 0.75
        config.training.label_smoothing = 0
        eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config, prefix)

        hyper_params = {'x_groups': [
            # Without previous beat
            [DatasetConfig.categorical, DatasetConfig.audio, DatasetConfig.regression],
            # Without ActionVec information
            [['prev_word_id'], DatasetConfig.categorical, DatasetConfig.audio, DatasetConfig.regression],
            [['prev_word_id'], DatasetConfig.categorical, DatasetConfig.audio, ],
            [['prev_word_id'], DatasetConfig.categorical, DatasetConfig.regression],
            [['prev_word_id'], DatasetConfig.audio, DatasetConfig.regression],
            [['prev_word_id'], ],
            # Without one data stream
            [['prev_word_vec'], ],
            [['prev_word_vec'], DatasetConfig.categorical, DatasetConfig.audio, DatasetConfig.regression],
            [['prev_word_vec'], DatasetConfig.categorical, DatasetConfig.audio, ],
            [['prev_word_vec'], DatasetConfig.categorical, DatasetConfig.regression],
            [['prev_word_vec'], DatasetConfig.audio, DatasetConfig.regression],
            # Give it redundant inputs
            [['prev_word_vec', 'prev_word_id'], DatasetConfig.categorical, DatasetConfig.audio,
             DatasetConfig.regression],
            [['prev_word_vec', 'prev_word_id'], DatasetConfig.beat_elements_previous_prediction,
             DatasetConfig.categorical, DatasetConfig.audio, DatasetConfig.regression],
            [['prev_word_vec', ], DatasetConfig.beat_elements_previous_prediction, DatasetConfig.categorical,
             DatasetConfig.audio, DatasetConfig.regression],
        ]}
        config = Config()
        config.training.mixup_alpha = 0.5
        config.training.label_smoothing = 0.5
        eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config, prefix)

        hyper_params = {'model_size': [1024, 768, 512, 384, 256, 128, 64, ]}
        config = Config()
        config.training.mixup_alpha = 0.5
        config.training.label_smoothing = 0.5
        eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config, prefix)

        hyper_params = {'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, ]}
        config = Config()
        config.training.mixup_alpha = 0.5
        config.training.label_smoothing = 0.5
        eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config, prefix)

        hyper_params = {'cnn_repetition': range(5)}
        config = Config()
        eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config, prefix)

        hyper_params = {'lstm_repetition': range(5)}
        config = Config()
        eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config, prefix)

        hyper_params = {'dense_repetition': range(5)}
        config = Config()
        eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config, prefix)
        pass


if __name__ == '__main__':
    main()
