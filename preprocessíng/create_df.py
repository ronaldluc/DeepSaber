import json
import os

import numpy as np
import pandas as pd


def create_ndarr(data: pd.DataFrame) -> np.array:
    output_shape = (3, 4, 2, 9)  # width × height × color × rotation (last == any)
    arr = np.zeros(output_shape)

    for index, row in data.iterrows():
        arr[row['_lineLayer'], row['_lineIndex'], row['_type'], row['_cutDirection']] = 1
    return arr


def create_dataframe(path: str, name: str) -> pd.DataFrame:
    os.chdir(path)

    with open(os.path.join(path, name)) as json_data:
        data = json.load(json_data)

    df = pd.DataFrame(data['_notes'])

    # - Most of the blocks appear precisely on the beats.
    # - Some are on half beats.
    # - From Expert and harder there are even triolas and wierd stuff

    # Round to 2 decimal places for normalization for block alignment
    df['_time'] = round(df['_time'], 2)

    out_df = pd.DataFrame(df.groupby('_time').apply(create_ndarr), columns=['output'])

    out_df['prev'] = 0
    last_time = 0

    for i, row in out_df.iterrows():
        out_df.loc[i, 'prev'] = i - last_time
        last_time = i

    out_df['next'] = out_df['prev'].shift(periods=-1).fillna(0)

    return out_df


if __name__ == '__main__':
    data_path = '/home/ron/Documents/programovani/projekty/deepSaber/data'

    hard = 'Kneel Before Me/Normal.json'
    extreme_plus = '[A]ddiction/ExpertPlus.json'

    df = create_dataframe(data_path, extreme_plus)
    print(df)
