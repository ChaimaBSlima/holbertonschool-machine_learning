#!/usr/bin/env python3
""" Task 3: 3. Create a Layer with L2 Regularization """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow 2.x dense layer with L2 regularization.

    Parameters:
    prev (tensor): The input tensor to the layer.
    n (int): The number of units (neurons) in the dense layer.
    activation (function): The activation function for the layer.
    lambtha (float): The L2 regularization parameter (lambda).

    Returns:
    tensor:
    The output of the dense layer after applying the activation function.
    """
    layer_weight = tf.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg"))
    L2_regularization = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=layer_weight,
                                  kernel_regularizer=L2_regularization)
    return layer(prev)
