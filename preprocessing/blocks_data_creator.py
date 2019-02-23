import os
from pprint import pprint

from typing import List
from preprocessing.json_loader import json_to_blockmask


def generate_blocks(path: str, difficulties: List[str]):
    difficulties = [f'{d}.json' for d in difficulties]
    stats = {'processed': 0, 'failed': 0}

    for root, _, files in os.walk(path, topdown=False):
        for name in files:
            if name in difficulties:
                try:
                    df = json_to_blockmask(os.path.join(root, name))
                    os.makedirs(os.path.join(root, 'blocks'), exist_ok=True)
                    df.to_pickle(os.path.join(
                        root, 'blocks', f'{name[:-5]}.pkl'))
                except KeyError as e:
                    stats['failed'] += 1
                    print(f'Failed on {root}/{name} | {str(e)[:20]}')
                    continue
                stats['processed'] += 1

    pprint(stats)


if __name__ == '__main__':
    generate_blocks('../data', ['Hard', 'Expert', 'ExpertPlus'])
