# %%

import numpy as np
import pandas as pd
import itertools
import json

df = pd.read_csv(
    "postprocessing/generated.csv",
    names=[
        '_time',
        '_lineLayer0',
        '_lineIndex0',
        '_cutDirection0',
        '_lineLayer1',
        '_lineIndex1',
        '_cutDirection1',
    ],
    header=0
)

df

# %%

twins = [(
    (
        row['_time'] * (107 / 60),
        int(row['_lineLayer0']),
        int(row['_lineIndex0']),
        int(row['_cutDirection0']),
        1
    ),
    (
        row['_time'] * (107 / 60),
        int(row['_lineLayer1']),
        int(row['_lineIndex1']),
        int(row['_cutDirection1']),
        0
    )
) for idx, row in df.iterrows()
]

twins = [
    [t[0]] if (t[0][1], t[0][2]) == (t[1][1], t[1][2]) else t
    for t in twins
]


df = pd.DataFrame(list(itertools.chain.from_iterable(twins)), columns=[
    '_time',
    '_lineLayer',
    '_lineIndex',
    '_cutDirection',
    '_type'
])

df

# %%
with open("postprocessing/song_template/Expert.json") as f:
    json_song = json.load(f)

    json_song['_notes'] = df.to_dict('records')

    with open('data/Torture Dance.1/Expert.json', 'w') as f:
        json.dump(json_song, f)
