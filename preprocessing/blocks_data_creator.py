import os

from typing import List
from json_loader import json_to_blockmask


def generate_blocks(path: str, difficulties: List[str]):
    difficulties = [f'{d}.json' for d in difficulties]

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name in difficulties:
                df = json_to_blockmask(os.path.join(root, name))
                os.makedirs(os.path.join(root, 'blocks'), exist_ok=True)
                df.to_pickle(os.path.join(root, 'blocks', f'{name[:-5]}.pkl'))


if __name__ == '__main__':
    generate_blocks('../data', ['Hard', 'Expert', 'ExpertPlus'])
