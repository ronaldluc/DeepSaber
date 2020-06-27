import math

from tensorflow import keras
from tensorflow.python.framework import ops, constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.experimental.FlatCosAnnealSchedule")
class FlatCosAnnealSchedule(keras.experimental.CosineDecay):
    """A LearningRateSchedule that uses a flat cosine decay schedule.

  See fastAI discussion https://forums.fast.ai/t/fastai-v2-callbacks-learner-optimizer/53519

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

    def __init__(
            self,
            decay_start,
            initial_learning_rate,
            decay_steps,
            alpha=0.0,
            name=None):
        """If Applies cosine decay to the learning rate.

    Args:
      decay_start: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to wait before performing cosine annealing.
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a
        Python number. The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of initial_learning_rate.
      name: String. Optional name of the operation.  Defaults to 'CosineDecay'.
    """
        super(FlatCosAnnealSchedule, self).__init__(initial_learning_rate,
                                                    decay_steps,
                                                    alpha,
                                                    name)
        self.decay_start = decay_start

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "FlatCosAnnealSchedule"):
            initial_learning_rate = ops.convert_to_tensor_v2(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_start = math_ops.cast(self.decay_start, dtype)
            decay_steps = math_ops.cast(self.decay_steps, dtype) - decay_start

            global_step_recomp = math_ops.cast(step, dtype) - decay_start
            global_step_recomp = math_ops.maximum(constant_op.constant(0.0), global_step_recomp)
            global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
            completed_fraction = global_step_recomp / decay_steps
            cosine_decayed = 0.5 * (1.0 + math_ops.cos(
                constant_op.constant(math.pi) * completed_fraction))

            decayed = (1 - self.alpha) * cosine_decayed + self.alpha
            res = math_ops.multiply(initial_learning_rate, decayed)
            return res

    def get_config(self):
        return {
            "decay_start": self.decay_start,
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name
        }
