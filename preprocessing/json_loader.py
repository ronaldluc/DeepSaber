import json
import os

import numpy as np
import pandas as pd


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


def compute_true_time(df: pd.DataFrame, bpm_df: pd.DataFrame, start_bpm: float) -> pd.Series:
    true_time = pd.Series(data=[0.0]*len(df))

    current_bpm = start_bpm
    current_beat = 0.0
    current_time = 0.0
    next_event_index = 0

    def advance_time(obj):
        nonlocal current_time, current_beat
        current_time += (obj['_time'] - current_beat) * (60.0 / current_bpm)
        current_beat = obj['_time']

    for i in range(len(df)):
        next_block = df.iloc[i]

        # Apply BPM changes
        while next_event_index < len(bpm_df) and bpm_df.iloc[next_event_index]['_time'] < next_block['_time']:
            next_event = bpm_df.iloc[next_event_index]
            advance_time(next_event)
            current_bpm = next_event['_value']
            next_event_index += 1

        advance_time(next_block)
        true_time.loc[i] = current_time
    return true_time


def json_to_blockmask(path: str) -> pd.DataFrame:
    with open(path) as json_data:
        data = json.load(json_data)

    # Load notes
    df = pd.DataFrame(data['_notes'])

    # Throw away bombs
    df = df.loc[df['_type'] != 3]

    # Round to 2 decimal places for normalization for block alignment
    df['_time'] = round(df['_time'], 2)

    # Load event times dataframe
    bpm_df = pd.DataFrame(
        data['_events'],
        columns=['_time', '_value', '_type']
    )
    bpm_df = bpm_df.loc[
        bpm_df.apply(lambda x: '_type' in x and x['_type'] == 14, axis=1)
    ].filter(items=['_time', '_value'])
    bpm_df['_value'] /= 1000

    # Compute actual time in seconds, not beats
    df['_time'] = compute_true_time(df, bpm_df, data["_beatsPerMinute"])

    out_df = pd.DataFrame(
        df.groupby('_time').apply(data_to_blockmask),
        columns=['output']
    )

    out_df['time'] = out_df.index
    out_df['prev'] = out_df['time'].diff().fillna(out_df['time'])
    out_df['next'] = out_df['prev'].shift(periods=-1).fillna(0)
    out_df.drop(labels='time', axis=1)

    # Indexes: _time
    # Cols: outputs (3 × 4 × 2 × 9 dim np.array), time, prev, next
    return out_df


if __name__ == '__main__':
    data_path = '../data'

    normal = '[A]ddiction/Expert.json'

    expert = 'One More Time/Expert.json'

    rasputin = 'Rasputin/Hard.json'

    df = json_to_blockmask(os.path.join(data_path, rasputin))
    print(df)
