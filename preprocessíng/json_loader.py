import json
import os

import numpy as np
import pandas as pd


def json_to_blockmask(path: str) -> pd.DataFrame:
    with open(path) as json_data:
        data = json.load(json_data)

    df = pd.DataFrame(data['_notes'])

    # - Most of the blocks appear precisely on the beats.
    # - Some are on half beats.
    # - From Expert and harder there are even triolas and wierd stuff

    # Round to 2 decimal places for normalization for block alignment
    df['_time'] = round(df['_time'], 2)

    # Throw away bombs
    df = df.loc[df['_type'] != 3]

    def data_to_blockmask(data: pd.DataFrame) -> np.array:
        # width × height × color × rotation (last == any)
        output_shape = (3, 4, 2, 9)
        arr = np.zeros(output_shape)

        for _, row in data.iterrows():
            arr[
                row['_lineLayer'],
                row['_lineIndex'],
                row['_type'],
                row['_cutDirection']
            ] = 1
        return arr

    out_df = pd.DataFrame(
        df.groupby('_time').apply(data_to_blockmask),
        columns=['output']
    )

    out_df['time'] = out_df.index
    out_df['prev'] = out_df['time'].diff().fillna(out_df['time'])
    out_df['next'] = out_df['prev'].shift(periods=-1).fillna(0)

    # Indexes: _time
    # Cols: outputs (3 × 4 × 2 × 9 dim np.array), prev, next
    return out_df


if __name__ == '__main__':
    data_path = './data'

    normal = 'One More Time/Normal.json'
    expert = 'One More Time/Expert.json'

    df = json_to_blockmask(os.path.join(data_path, normal))
    print(df)
