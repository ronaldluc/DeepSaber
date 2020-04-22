import json

import numpy as np
import pandas as pd


def blockmask_to_data(blockmask: np.array, time: float) -> pd.DataFrame:
    return pd.DataFrame(
        data=((time, *coords)
              for coords, value in np.ndenumerate(blockmask)
              if value == 1
              ),
        columns=['_time', '_lineLayer', '_lineIndex', '_type', '_cutDirection']
    )


def blockmasks_to_json(blockmasks: pd.DataFrame, path: str):
    with open("postprocessing/song_template/Expert.json") as f:
        json_song = json.load(f)

    json_song['_notes'] = pd.concat(
        blockmask_to_data(bm, time)
        for (time, bm) in blockmasks["output"].iteritems()
    ).to_dict('records')

    with open(path, 'w') as f:
        json.dump(json_song, f)


def save_song_info():
    pass
