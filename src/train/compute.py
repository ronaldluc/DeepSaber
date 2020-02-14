import numpy as np
import pandas as pd
import tensorflow as tf


def to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn beat elements dimensions into one-hot-encoding
    :param df: beat elements
    :return: updated beat elements
    """
    dim_max = {'_lineLayer': 3, '_lineIndex': 4, '_cutDirection': 9}
    for col, dim in dim_max.items():
        # df[col] = tf.keras.utils.to_categorical(df[col], dim).tolist()
        num = tf.keras.utils.to_categorical(df[col], dim, dtype=np.int8)
        flatten = np.split(num.flatten(), len(df.index))
        df[col] = flatten
    return df