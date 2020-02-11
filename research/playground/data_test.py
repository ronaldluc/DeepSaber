import json
import os
from pprint import pprint

import pandas as pd


def explore_structure():
    difficulties = [f'{d}.dat' for d in ['Hard', 'Expert', 'ExpertPlus']]
    path = '../../data/new_dataformat'
    paths = []
    for root, _, files in os.walk(path, topdown=False):
        for name in files:
            if name in difficulties:
                paths.append(os.path.join(root, name))
    for path in paths:
        with open(path) as json_data:
            data = json.load(json_data)
            print(data.keys())
            if '_BPMChanges' in data:
                if len(data['_BPMChanges']) > 0:
                    print('BPMChanges!')
                    print(path)
                    print(data['_BPMChanges'])
                    bpm_df = pd.DataFrame(
                        data['_BPMChanges'],
                        columns=['_time', '_BPM']
                    ).sort_values('_time').rename(columns={'_BPM': '_value'})

                    bpm_df = bpm_df.append(bpm_df, ignore_index=True)
                    print(bpm_df)
            else:
                print('-' * 42)
                pprint(data)
                print('-' * 42)


def explore_info():
    path = '../../data/new_dataformat'
    paths = []
    for root, _, files in os.walk(path, topdown=False):
        for name in files:
            if name == 'info.dat':
                paths.append(os.path.join(root, name))

    print(paths)
    for path in paths:
        with open(path) as json_data:
            data = json.load(json_data)
            if '_songTimeOffset' in data:
                if int(data['_songTimeOffset']) > 0:
                    pprint(data)
            else:
                print('FAIL ' * 42)
                pprint(data)
                print('-' * 42)


def test_bpm():
    folder = '3ca4'
    path = f'../../data/new_dataformat/{folder}/ExpertPlus.dat'
    path_info = f'../../data/new_dataformat/{folder}/info.dat'
    with open(path) as json_data:
        data = json.load(json_data)

        # print(data.keys())
        if '_BPMChanges' in data:
            if len(data['_BPMChanges']) > 0:
                # pprint(data)
                print(data['_BPMChanges'])

        max_time = max([x['_time'] for x in data['_notes']])

        print(max_time)
    with open(path_info) as info_data:
        data = json.load(info_data)
        bpm = data['_beatsPerMinute']

        print(f'Song length {max_time // bpm} minutes {max_time / bpm * 60 % 60} seconds')


if __name__ == '__main__':
    explore_structure()
    # explore_info()
    # test_bpm()

#     Findings:
#       No files use `_songTimeOffset`.
#       Length of the song matches `_time / bpm`
#       Some songs include `_BMPChanges`
#       Some songs do not have key `_BMPChanges`

