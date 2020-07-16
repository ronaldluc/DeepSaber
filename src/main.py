import gc
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from predict.api import generate_complete_beatmaps
from process.api import load_datasets, generate_datasets, create_song_list
from train.callbacks import create_callbacks
from train.metrics import Perplexity
from train.model import save_model, get_architecture_fn
from train.sequence import BeatmapSequence
from utils.functions import dataset_stats
from utils.types import Config, Timer


def main():
    timer = Timer()

    seed = 43  # random, non-fine tuned seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    config = Config()

    base_folder = config.base_data_folder

    song_folders = create_song_list(config.dataset.beat_maps_folder)

    # total = len(song_folders)
    # print(f'Found {total} folders')

    config.dataset.storage_folder = base_folder / 'new_datasets_config_test'
    # config.dataset.storage_folder = base_folder / 'old_datasets'
    # config.dataset.storage_folder = base_folder / 'new_datasets'
    config.dataset.storage_folder = base_folder / 'test_datasets'
    config.audio_processing.use_cache = False

    # #### TODO: DELETE
    config.training.data_split = (0.0, 1.0, 1.0, 1.0,)
    # data_folder = Path('../../data')
    # config.base_data_folder = data_folder
    base_folder = config.base_data_folder
    config.dataset.beat_maps_folder = base_folder / 'evaluation_dataset' / 'beat_sage'
    print(config.dataset.beat_maps_folder)
    config.dataset.storage_folder = base_folder / 'beatsage_datasets'
    song_folders = create_song_list(config.dataset.beat_maps_folder)
    #########

    generate_datasets(song_folders, config)

    train, val, test = load_datasets(config)
    timer('Loaded datasets', 5)

    # Ensure this song is excluded from the training data for hand tasting
    train.drop(index='133b', inplace=True, errors='ignore')
    train.drop(index='Daddy - PSY', inplace=True, errors='ignore')
    dataset_stats(train)

    train_seq = BeatmapSequence(df=train, is_train=True, config=config)
    val_seq = BeatmapSequence(df=val, is_train=False, config=config)
    test_seq = BeatmapSequence(df=test, is_train=False, config=config)
    timer('Generated sequences', 5)

    # del train, val, test  # delete the data if experiencing RAM problems
    gc.collect()

    # keras.mixed_precision.experimental.set_policy('mixed_float16')
    model_path = base_folder / 'temp'
    model_path.mkdir(parents=True, exist_ok=True)

    train = True
    # train = False
    if train:
        model = get_architecture_fn(config)(train_seq, False, config)
        model.summary()

        callbacks = create_callbacks(train_seq, config)
        # callbacks = []

        model.fit(train_seq,
                  validation_data=val_seq,
                  callbacks=callbacks,
                  epochs=400,
                  verbose=2,
                  workers=10,
                  max_queue_size=16,
                  use_multiprocessing=False,
                  )
        timer('Trained model', 5)
        model.evaluate(test_seq)
        timer('Evaluated model', 5)

        save_model(model, model_path, train_seq, config)
        timer('Saved model', 5)

    stateful_model = keras.models.load_model(model_path / 'stateful_model.keras',
                                             custom_objects={'Perplexity': Perplexity})
    stateful_model.summary()
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
