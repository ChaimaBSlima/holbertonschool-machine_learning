#!/usr/bin/env python3
"""Task 2: 2. Identity Block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds a projection block as part of a residual network.

    Parameters
    A_prev : keras.layers.Layer
        The output from the previous layer, used as the input for the
        projection block.
    filters : list of int
        A list containing the number of filters for each convolutional layer
        in the block. The filters should be provided in the order
        [F11, F3, F12], where:
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
        The output is combined with the input through a shortcut path with
        downsampling.
    """
    F11, F3, F12 = filters
    c1 = K.layers.Conv2D(
        filters=F11, kernel_size=(
            1, 1), padding='same', strides=(
            1, 1), kernel_initializer=K.initializers.he_normal(
                seed=0))(A_prev)
    b1 = K.layers.BatchNormalization(axis=3)(c1)
    r1 = K.layers.Activation('relu')(b1)
    c2 = K.layers.Conv2D(
        filters=F3, kernel_size=(
            3, 3), padding='same', strides=(
            1, 1), kernel_initializer=K.initializers.he_normal(
                seed=0))(r1)
    b2 = K.layers.BatchNormalization(axis=3)(c2)
    r2 = K.layers.Activation('relu')(b2)
    c3 = K.layers.Conv2D(
        filters=F12, kernel_size=(
            1, 1), padding='same', strides=(
            1, 1), kernel_initializer=K.initializers.he_normal(
                seed=0))(r2)
    b3 = K.layers.BatchNormalization(axis=3)(c3)
    A = K.layers.Add()([b3, A_prev])
    A = K.layers.Activation('relu')(A)

    return A
