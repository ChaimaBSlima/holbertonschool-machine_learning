#!/usr/bin/env python3
""" Task 6:  6. Transition Layer """
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer for the DenseNet architecture.

    Parameters
    X : tensor
        Input tensor of shape (height, width, channels).
    nb_filters : int
        The number of filters before applying compression.
    compression : float
        The compression factor to reduce the number of filters.

    Returns
    X : tensor
        Output tensor after passing through the transition layer.
    nb_filters : int
        The number of filters after compression.
    """
    # implement He et. al initialization for the layers weights
    initializer = K.initializers.he_normal(seed=0)

    my_layer = K.layers.BatchNormalization()(X)
    my_layer = K.layers.ReLU()(my_layer)

    nb_filters = int(nb_filters * compression)

    my_layer = K.layers.Conv2D(filters=nb_filters,
                               kernel_size=1,
                               padding='same',
                               kernel_initializer=initializer,
                               )(my_layer)

    # Avg pooling layer with kernels of shape 2x2
    X = K.layers.AveragePooling2D(pool_size=2,
                                  padding='same')(my_layer)

    return X, nb_filters
