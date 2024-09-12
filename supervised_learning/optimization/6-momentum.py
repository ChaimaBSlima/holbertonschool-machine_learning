#!/usr/bin/env python3
""" Task 6: 6. Momentum Upgraded  """
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network in TensorFlow
    using the gradient descent with momentum optimization algorithm.

    Args:
        loss (tf.Tensor): The loss tensor of the network, typically calculated
            using a loss function such as cross-entropy or mean squared error.
        alpha (float): The learning rate, controlling the step size in the
            optimization process.
        beta1 (float): The momentum weight, controlling the contribution of
            previous gradients to the update, a value between 0 and 1.

    Returns:
        tf.Operation: The operation that applies the gradient descent with
        momentum optimization to minimize the loss.
    """
    optimizer = tf.compat.v1.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
