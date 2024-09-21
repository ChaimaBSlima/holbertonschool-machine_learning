#!/usr/bin/env python3
""" Task 2: 2. L2 Regularization Cost """
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the total cost of a neural network including L2 regularization.

    This function adds the L2 regularization losses of each layer to the
    given base cost of the model (e.g., cross-entropy loss).

    Parameters:
    cost (tensor): The original cost without L2 regularization.
    model (tf.keras.Model): The neural network model containing the layers
                            with L2 regularization.

    Returns:
    tensor:
    A tensor containing the total cost including L2 regularization losses.
    """

    l2_reg_loss = list()
    for layer in model.layers:
        l2_reg_loss.append(tf.reduce_sum(layer.losses) + cost)

    return tf.stack(l2_reg_loss[1:])
