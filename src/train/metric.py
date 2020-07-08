import logging

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.metrics import cosine_similarity, MeanMetricWrapper
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.util import dispatch

from utils.types import Config


def create_metrics(is_train, config: Config):
    if is_train:
        return ['acc',
                tf.keras.metrics.CategoricalAccuracy(),
                Perplexity(),
                ]
    print('Create metrics not train')
    return ['acc', ]


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


class PerplexityMetric(K.metrics.Metric):
    """
    USAGE NOTICE: this metric accepts only logits for now (i.e. expect the same behaviour
        as from tf.keras.losses.SparseCategoricalCrossentropy with the a provided argument "from_logits=True",
        here the same loss is used with "from_logits=True" enforced so you need to provide it in such a format)
    METRIC DESCRIPTION:
    Popular metric for evaluating language modelling architectures.
    More info: http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf.
    DISCLAIMER: Based on Gregorgeous github Gist: https://gist.github.com/Gregorgeous/dbad1ec22efc250c76354d949a13cec3
    Original function created by Kirill Mavreshko in https://github.com/kpot/keras-transformer/blob/b9d4e76c535c0c62cadc73e37416e4dc18b635ca/example/run_gpt.py#L106.
    """

    def __init__(self, name='perplexity', **kwargs):
        super(PerplexityMetric, self).__init__(name=name, **kwargs)
        # self.cross_entropy = K.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.cross_entropy = K.losses.CategoricalCrossentropy(from_logits=False, reduction='none')
        self.perplexity = self.add_weight(name='tp', initializer='zeros')

    # Consider uncommenting the decorator for a performance boost (?)
    @tf.function
    def _calculate_perplexity(self, real, pred):
        # The next 4 lines zero-out the padding from loss calculations,
        # this follows the logic from: https://www.tensorflow.org/beta/tutorials/text/transformer#loss_and_metrics
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.cross_entropy(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        # Calculating the perplexity steps:
        step1 = K.backend.mean(loss_, axis=-1)
        step2 = K.backend.exp(step1)
        perplexity = K.backend.mean(step2)

        return perplexity

    def update_state(self, y_true, y_pred, sample_weight=None):
        # TODO:FIXME: handle sample_weight !
        if sample_weight is not None:
            logging.log(logging.WARNING,
                        "Provided 'sample_weight' argument to the perplexity metric. "
                        "Currently this is not handled and won't do anything differently.")
        perplexity = self._calculate_perplexity(y_true, y_pred)
        # Remember self.perplexity is a tensor (tf.Variable), so using simply "self.perplexity = perplexity" will
        # result in error because of mixing EagerTensor and Graph operations
        self.perplexity.assign_add(perplexity)

    def result(self):
        return self.perplexity

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.perplexity.assign(0.)


@dispatch.add_dispatch_support
def calculate_perplexity(y_true, y_pred, from_logits=False, label_smoothing=0):
    """
    Based on Gregorgeous github Gist: https://gist.github.com/Gregorgeous/dbad1ec22efc250c76354d949a13cec3
    """
    # The next 4 lines zero-out the padding from loss calculations,
    # this follows the logic from: https://www.tensorflow.org/beta/tutorials/text/transformer#loss_and_metrics
    # mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = K.losses.categorical_crossentropy(y_true, y_pred, from_logits=from_logits, label_smoothing=label_smoothing)
    # mask = tf.cast(mask, dtype=loss_.dtype)
    # loss_ *= mask
    # Calculating the perplexity steps:
    step1 = K.backend.mean(loss_, axis=-1)
    step2 = K.backend.exp(step1)
    perplexity = K.backend.mean(step2)

    return perplexity


class Perplexity(LossFunctionWrapper):

    def __init__(self,
                 from_logits=False,
                 label_smoothing=0,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='perplexity'):
        """Initializes `Perplexity` instance.

    Args:
      from_logits: Whether `y_pred` is expected to be a logits tensor. By
        default, we assume that `y_pred` encodes a probability distribution.
        **Note - Using from_logits=True is more numerically stable.**
      label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
        meaning the confidence on label values are relaxed. e.g.
        `label_smoothing=0.2` means that we will use a value of `0.1` for label
        `0` and `0.9` for label `1`"
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to 'perplexity'.
    """
        super(Perplexity, self).__init__(
            calculate_perplexity,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing)
