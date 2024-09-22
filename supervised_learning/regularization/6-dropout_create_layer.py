#!/usr/bin/env python3
""" Task 6 : 6. Create a Layer with Dropout """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a dense layer with dropout regularization.

    Parameters:
    prev (tensor):
    The input tensor to the layer.
    n (int):
    The number of units (neurons) in the dense layer.
    activation (function):
    The activation function to apply (e.g., tf.nn.relu).
    keep_prob (float):
    The probability of keeping a unit active during dropout (between 0 and 1).
    training (bool):
    A flag indicating whether the model is being trained.
    Dropout is only applied during training.

    Returns:
    tensor:
    The output tensor after applying the dense layer and dropout
    (if training is True).
    """

    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer)(prev)

    if training:
        layer = tf.nn.dropout(layer, rate=1 - keep_prob)

    return layer
