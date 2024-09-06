#!/usr/bin/env python3
""" Task 2: 2. Optimize """
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Configures the Adam optimizer for a Keras model
    using categorical crossentropy loss and accuracy metrics.

    Parameters:
    network : keras.Model
        The Keras model to optimize.
    alpha : float
        The learning rate for the Adam optimizer.
    beta1 : float
        The exponential decay rate for the first
        moment estimates (Adam parameter).
    beta2 : float
        The exponential decay rate for the second
        moment estimates (Adam parameter).

    Returns:
    None
    """

    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return None
