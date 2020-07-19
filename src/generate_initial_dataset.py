"""
This script is used to kick-start the notebooks.
It computes the audio features for the dataset and creates the initial train, val, test
pickle DataFrames without AVS and word bijection representations.
To generate all action representation, afterwards run `src/notebooks/create_action_embeddings.ipynb`.
"""

import random

import numpy as np

from process.api import load_datasets, create_song_list, generate_datasets
from utils.types import Config, Timer


def main():
    timer = Timer()

    seed = 43  # random, non-fine tuned seed
    np.random.seed(seed)
    random.seed(seed)
    config = Config()

    base_folder = config.base_data_folder

    # To generate full dataset
    song_folders = create_song_list(config.dataset.beat_maps_folder)
    # config.dataset.storage_folder = base_folder / 'old_datasets'
    config.dataset.storage_folder = base_folder / 'new_datasets'

    # To generate test dataset
    # song_folders = create_song_list(config.dataset.beat_maps_folder)[:100]
    # config.dataset.storage_folder = base_folder / 'test_datasets'
    config.audio_processing.use_cache = False  # The audio features need to be computed the first time
    config.use_multiprocessing = True  # since TF is not imported

    total = len(song_folders)
    print(f'Found {total} folders')

    generate_datasets(song_folders, config)

    # Test loading datasets
    train, val, test = load_datasets(config)
    timer('Loaded datasets', 5)


if __name__ == '__main__':
    main()
