#!/usr/bin/env python3
""" Task 12: 12. Learning Rate Decay Upgraded  """
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Updates the learning rate using inverse time decay in TensorFlow.

    Args:
        alpha:
            The original learning rate.
        decay_rate:
            The rate at which the learning rate decays.
        decay_step:
            The number of passes of gradient descent after which
            the learning rate should decay.

    Returns:
        Tensor: The decayed learning rate as a TensorFlow operation.
    """

    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)
