#!/usr/bin/env python3
""" Task 8: 8. RMSProp Upgraded """
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates the RMSProp optimization operation for a neural network
    using TensorFlow's Keras optimizer.

    Args:
        alpha (float):
            The learning rate for the optimizer.
        beta2 (float):
            The decay factor (rho) used to compute the moving
            average of squared gradients.
        epsilon (float):
            A small value added to the denominator to avoid division by zero.

    Returns:
        tf.keras.optimizers.RMSprop:
            The RMSProp optimizer instance configured
            the specified parameters.
    """
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                            rho=beta2, epsilon=epsilon)
    return optimizer
