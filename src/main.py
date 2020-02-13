import pandas as pd

from process.api import create_song_list, songs2dataset
from utils.functions import check_consistency
from utils.types import Config, Timer

if __name__ == '__main__':
    song_folders = create_song_list('../data')
    total = len(song_folders)

    config = Config()
    # config.audio_processing['use_cache'] = False

    timer = Timer()
    for phase, split in zip(['train', 'val', 'test'],
                            zip(config.training['data_split'],
                                config.training['data_split'][1:])
                            ):
        print('\n', '=' * 100, sep='')
        print(f'Processing {phase}')
        split_from = int(total * split[0])
        split_to = int(total * split[1])
        result_path = f'../data/{phase}_beatmaps.pkl'

        df = songs2dataset(song_folders[split_from:split_to], config=config)
        timer(f'Created {phase} dataset', 1)

        check_consistency(df)

        df.to_pickle(result_path)
        timer(f'Pickled {phase} dataset', 1)

