import gc
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from predict.api import generate_complete_beatmaps
from process.api import create_song_list, songs2dataset
from train.callbacks import create_callbacks
from train.model import create_model
from train.sequence import BeatmapSequence
from utils.functions import check_consistency
from utils.types import Config, Timer


def generate_datasets(song_folders, config: Config):
    timer = Timer()
    for phase, split in zip(['train', 'val', 'test'],
                            zip(config.training.data_split,
                                config.training.data_split[1:])
                            ):
        print('\n', '=' * 100, sep='')
        print(f'Processing {phase}')
        total = len(song_folders)
        split_from = int(total * split[0])
        split_to = int(total * split[1])
        result_path = config.dataset.storage_folder / f'{phase}_beatmaps.pkl'

        df = songs2dataset(song_folders[split_from:split_to], config=config)
        timer(f'Created {phase} dataset', 1)

        check_consistency(df)

        config.dataset.storage_folder.mkdir(parents=True, exist_ok=True)
        df.to_pickle(result_path, protocol=4)   # Protocol 4 for Python 3.6/3.7 compatibility
        timer(f'Pickled {phase} dataset', 1)


def load_datasets(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return [pd.read_pickle(config.dataset.storage_folder / f'{phase}_beatmaps.pkl') for phase in
            ['train', 'val', 'test']]


def dataset_stats(df: pd.DataFrame):
    print(df)
    group_over = ['name', 'difficulty', 'snippet', 'time', ]
    for end_index in range(1, len(group_over) + 1):
        print(f"{df.groupby(group_over[:end_index]).ngroups:9} {' × '.join(group_over[:end_index])}")


def list2numpy(batch, col_name, groupby=('name')):
    return np.array(batch.groupby(list(groupby))[col_name].apply(list).to_list())


def create_training_data(X, groupby, config: Config):
    X_cols = config.dataset.audio
    y_cols = config.dataset.beat_elements
    return [list2numpy(X, col, groupby) for col in X_cols], \
           [list2numpy(X, col, groupby) for col in y_cols]


def main():
    tf.random.set_seed(43)
    np.random.seed(43)
    random.seed(43)

    base_folder = Path('../data')
    song_folders = create_song_list(base_folder / 'human_beatmaps')
    total = len(song_folders)
    print(f'Found {total} folders')

    config = Config()
    config.dataset.storage_folder = base_folder / 'full_datasets'
    config.dataset.storage_folder = base_folder / 'new_datasets'
    # config.dataset.storage_folder = base_folder / 'test_datasets'
    # config.audio_processing.use_cache = False

    # generate_datasets(song_folders, config)

    train, val, test = load_datasets(config)
    print(train.columns)

    # Ensure this song is excluded from the training data for hand tasting
    train.drop(index='133b', inplace=True, errors='ignore')
    # dataset_stats(train)

    train_seq = BeatmapSequence(df=train, is_train=config.training.use_mixup, config=config)
    val_seq = BeatmapSequence(df=val, is_train=False, config=config)
    test_seq = BeatmapSequence(df=test, is_train=False, config=config)

    del train, val, test
    gc.collect()

    # keras.mixed_precision.experimental.set_policy('mixed_float16')
    model_path = base_folder / 'temp'
    model_path.mkdir(parents=True, exist_ok=True)

    train = True
    train = False
    if train:
        model = create_model(train_seq, False, config)
        model.summary()

        callbacks = create_callbacks(train_seq, config)
        # callbacks = []

        timer = Timer()

        model.fit(train_seq,
                  validation_data=val_seq,
                  callbacks=callbacks,
                  epochs=80,
                  verbose=2,
                  workers=10,
                  max_queue_size=16,
                  use_multiprocessing=False,
                  )
        timer('Training ')

        save_model(model, model_path, train_seq, config)

    stateful_model = keras.models.load_model(model_path / 'stateful_model.keras')
    print('Evaluation')

    beatmap_folder = base_folder / 'human_beatmaps' / 'new_dataformat' / '133b'

    output_folder = base_folder / 'testing' / 'generated_songs'

    stateful_model.summary()
    generate_complete_beatmaps(beatmap_folder, output_folder, stateful_model, config)


def debug_model(model: keras.Model):
    for layer in model.layers:
        shapes = [x.shape for x in layer.weights]
        print(f'{layer.name:12}: {shapes}')
    model.summary()


def save_model(model, model_path, train_seq, config):
    keras.mixed_precision.experimental.set_policy('float32')
    config.training.batch_size = 1
    stateful_model = create_model(train_seq, True, config)
    plain_model = keras.Model(model.inputs, model.outputs)  # drops non-serializable metrics, etc.
    stateful_model.set_weights(plain_model.get_weights())
    plain_model.save(model_path / 'model.keras')
    stateful_model.save(model_path / 'stateful_model.keras')
    return stateful_model


if __name__ == '__main__':
    main()
