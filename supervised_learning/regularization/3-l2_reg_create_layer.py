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
    reg = tf.keras.regularizers.L2(l2=lambtha)
    init = tf.keras.initializers.VarianceScaling(mode="fan_avg")

    layer = tf.keras.layers.Dense(units=n,
                                  activation=activation,
                                  kernel_initializer=init,
                                  kernel_regularizer=reg)
    return layer(prev)
