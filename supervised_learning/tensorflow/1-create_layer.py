#!/usr/bin/env python3
""" Task 1: 1. Layers """
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_layer(prev, n, activation):
    """
    Creates a fully connected layer in a neural network using TensorFlow.

    Parameters:
    prev : tf.Tensor
        The output tensor from the previous layer.
    n : int
        The number of neurons (nodes) in the layer to create.
    activation : callable or None
        The activation function to apply to the output of the layer.
        If None, no activation function is applied (i.e., linear activation).

    Returns:
    tf.Tensor
        The output tensor of the newly created layer after
        applying the activation function.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer, name='layer')

    output = layer(prev)

    return output
