#!/usr/bin/env python3
""" Task 2: 2. Identity Block """
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds a projection block as part of a residual network.
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=0)
    # 1x1 Convolution
    conv2d = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        strides=(1, 1),
        kernel_initializer=init
    )(A_prev)

    batch_normalization = K.layers.BatchNormalization()(conv2d)
    activation = K.layers.ReLU()(batch_normalization)

    conv2d_1 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        strides=(1, 1),
        kernel_initializer=init
    )(activation)

    batch_normalization_1 = K.layers.BatchNormalization()(conv2d_1)
    activation_1 = K.layers.ReLU()(batch_normalization_1)
    conv2d_2 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        strides=(1, 1),
        kernel_initializer=init
    )(activation_1)

    batch_normalization_2 = K.layers.BatchNormalization()(conv2d_2)

    add = K.layers.Add()([batch_normalization_2, A_prev])

    activation_2 = K.layers.ReLU()(add)

    return activation_2
