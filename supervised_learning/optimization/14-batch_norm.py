#!/usr/bin/env python3
""" Task 14: 14. Batch Normalization Upgraded """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev (tf.Tensor):
            The activated output of the previous layer.
        n (int):
            The number of nodes in the layer to be created.
        activation (function or None):
            The activation function to apply on the output
            of the layer. If None, no activation is applied.

    Returns:
        tf.Tensor:
            The output of the batch normalization layer,
            with activation applied if specified.
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, kernel_initializer=init)
    z = layer(prev)
    gamma = tf.Variable(1., trainable=True)
    beta = tf.Variable(0., trainable=True)
    mean = tf.math.reduce_mean(z, axis=0)
    var = tf.math.reduce_variance(z, axis=0)
    epsilon = 1e-8
    normalized = tf.nn.batch_normalization(
        z, mean, var, beta, gamma, epsilon)
    return activation(normalized)
