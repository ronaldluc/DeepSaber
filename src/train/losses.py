from tensorflow import keras
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.util import dispatch


@dispatch.add_dispatch_support
def calculate_perplexity(y_true, y_pred, from_logits=False, label_smoothing=0):
    """
    Based on Gregorgeous github Gist: https://gist.github.com/Gregorgeous/dbad1ec22efc250c76354d949a13cec3
    Does not support masks.
    """
    # The next 4 lines zero-out the padding from loss calculations,
    # this follows the logic from: https://www.tensorflow.org/beta/tutorials/text/transformer#loss_and_metrics
    # mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=from_logits,
                                                  label_smoothing=label_smoothing)
    step1 = keras.backend.mean(loss_, axis=-1)
    step2 = keras.backend.exp(step1)
    perplexity = keras.backend.mean(step2)

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
