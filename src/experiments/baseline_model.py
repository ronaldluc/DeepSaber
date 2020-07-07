from experiments.compute import eval_hyperparams, init_test

from utils.types import Config


def main():
    base_folder, return_list, test, timer, train, val = init_test()
    prefix = 'baseline_'

    for repetition in range(7):
        hyper_params = {'model_size': [1024, 768, 512, 384, 256, 128, 64, ]}
        config = Config()
        config.training.model_type = 'baseline'
        config.training.cnn_repetition = 0
        config.training.lstm_repetition = 1
        config.training.dense_repetition = 0
        config.training.dropout = 0
        config.training.initial_learning_rate = 0.001
        config.training.batch_size = 128
        config.training.label_smoothing = 0
        config.training.mixup_alpha = 0
        config.training.l2_regularization = 0
        config.training.x_groups = [['prev_word_id'], ]
        config.training.y_groups = [['word_id'], ]
        eval_hyperparams(base_folder, timer, hyper_params, return_list, train, val, test, config, prefix)
        pass


if __name__ == '__main__':
    main()
