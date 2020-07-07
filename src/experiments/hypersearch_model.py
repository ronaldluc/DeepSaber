import kerastuner as kt
import tensorflow as tf

from experiments.compute import init_test
from predict.api import generate_complete_beatmaps
from train.callbacks import create_callbacks
from train.metric import Perplexity
from train.model import save_model, \
    get_architecture_fn
from train.sequence import BeatmapSequence
from utils.types import Config


def main():
    base_folder, return_list, test, timer, train, val = init_test()
    prefix = 'hyper_'
    config = Config()
    model_path = base_folder / 'temp'

    find_model = True
    # find_model = False
    train_model = True
    train_model = False
    if find_model:
        train_seq = BeatmapSequence(df=train, is_train=True, config=config)
        val_seq = BeatmapSequence(df=val, is_train=False, config=config)
        test_seq = BeatmapSequence(df=test, is_train=False, config=config)

        config.training.model_type = 'multi_lstm_tune'

        tuner = kt.Hyperband(
            get_architecture_fn(config)(train_seq, False, config),
            objective=kt.Objective('val_acc', direction='max'),
            hyperband_iterations=2,
            max_epochs=50,
            factor=3,
            directory=base_folder / 'temp' / 'hyper_search',
            # project_name='TEST',
            project_name=f'{get_architecture_fn(config).__qualname__}5',
            overwrite=False,  # TODO: CAUTION!
        )
        tuner.search_space_summary()

        callbacks = create_callbacks(train_seq, config)

        tuner.search(x=train_seq,
                     validation_data=val_seq,
                     callbacks=callbacks,
                     epochs=50,
                     verbose=2,
                     workers=10,
                     max_queue_size=16,
                     use_multiprocessing=False,
                     )

        print(tuner.results_summary())
        # print(tuner.get_best_models(2)[0].summary())
        # print(tuner.get_best_models(2)[0].evaluate(test_seq))

    if train_model:
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
        for param, val in fixed_params.items():
            hp.Fixed(param, value=val)

        model = get_architecture_fn(config)(train_seq, False, config)(hp)
        model.summary()
        tf.keras.utils.plot_model(model, to_file=base_folder / 'temp' / 'model_architecture.png', show_shapes=True)
        model.fit(x=train_seq,
                  validation_data=val_seq,
                  callbacks=callbacks,
                  epochs=81,
                  verbose=2,
                  workers=10,
                  max_queue_size=16,
                  use_multiprocessing=False,
                  )

        model_path.mkdir(parents=True, exist_ok=True)

        save_model(model, model_path, train_seq, config, hp=hp)
        timer('Saved model', 5)

    stateful_model = tf.keras.models.load_model(model_path / 'stateful_model.keras',
                                                custom_objects={'Perplexity': Perplexity})

    timer('Loaded stateful model', 5)

    input_folder = base_folder / 'human_beatmaps' / 'new_dataformat'
    output_folder = base_folder / 'testing' / 'generated_songs'
    song_codes_to_gen = list(x for x in test.index.to_frame()["name"].unique()[:5])
    song_codes_to_gen = ['133b', ]
    print(song_codes_to_gen)
    for song_code in song_codes_to_gen:
        beatmap_folder = input_folder / song_code
        print(beatmap_folder)
        generate_complete_beatmaps(beatmap_folder, output_folder, stateful_model, config)
        timer('Generated beatmaps', 5)


if __name__ == '__main__':
    main()
