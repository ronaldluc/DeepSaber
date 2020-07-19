from tensorflow.python.keras.metrics import cosine_similarity, MeanMetricWrapper

from train.losses import calculate_perplexity
from utils.types import Config


def create_metrics(is_train, config: Config):
    if is_train:
        return []  # all are computed by the AVS model
    return ['acc', Perplexity()]


def compute_acc(res_dict):
    acc = [val for key, val in res_dict.items() if 'acc' in key]
    if len(acc) == 0:
        return 0.0
    return sum(acc) / len(acc)


def cosine_distance(*args, **kwards):
    return 1 - cosine_similarity(*args, **kwards)


class CosineDistance(MeanMetricWrapper):

    def __init__(self, name='cosine_distance', dtype=None, axis=-1):
        """Creates a `CosineSimilarity` instance.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      axis: (Optional) Defaults to -1. The dimension along which the cosine
        similarity is computed.
    """
        super(CosineDistance, self).__init__(cosine_distance, name, dtype=dtype, axis=axis)


class Perplexity(MeanMetricWrapper):
    def __init__(self, name='perplexity', dtype=None):
        super(Perplexity, self).__init__(calculate_perplexity, name, dtype=dtype)
