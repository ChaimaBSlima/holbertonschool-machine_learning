#!/usr/bin/env python3
""" Task 3: 3. Projection Block """
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as part of a residual network.

    Parameters
    A_prev : keras.layers.Layer
        The output from the previous layer, used as the input for
        the projection block.
    filters : list of int
        A list containing the number of filters for each convolutional layer
        in the block.
        The filters should be provided in the order [F11, F3, F12], where:
        - F11: Number of filters for the first 1x1 convolution.
        - F3: Number of filters for the 3x3 convolution.
        - F12: Number of filters for the second 1x1 convolution.
    s : int, optional
        The stride value for the first and shortcut 1x1 convolutions
        (default is 2).

    Returns
    output : keras.layers.Layer
        The output of the projection block after performing a series
        of convolutions, batch normalization, and activation operations.
        The output is combined with the input through a shortcut path
        with downsampling.
    """
    F11, F3, F12 = filters

    # implement He et. al initialization for the layers weights
    initializer = K.initializers.he_normal(seed=None)

    # Conv 1x1
    my_layer = K.layers.Conv2D(filters=F11,
                               kernel_size=(1, 1),
                               strides=(s, s),
                               padding='same',
                               kernel_initializer=initializer,
                               )(A_prev)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)
    my_layer = K.layers.Activation('relu')(my_layer)

    # Conv 3x3
    my_layer = K.layers.Conv2D(filters=F3,
                               kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=initializer,
                               )(my_layer)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)
    my_layer = K.layers.Activation('relu')(my_layer)

    # Conv 1x1
    my_layer = K.layers.Conv2D(filters=F12,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=initializer,
                               )(my_layer)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)

    # shotcut path
    # Conv 1x1
    short = K.layers.Conv2D(filters=F12,
                            kernel_size=(1, 1),
                            strides=(s, s),
                            padding='same',
                            kernel_initializer=initializer,
                            )(A_prev)

    short = K.layers.BatchNormalization(axis=3)(short)

    output = K.layers.Add()([my_layer, short])

    output = K.layers.Activation('relu')(output)

    return output
