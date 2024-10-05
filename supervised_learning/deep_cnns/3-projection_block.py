#!/usr/bin/env python3
""" Task 3: 3. Projection Block """
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as part of a residual network.
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    conv0 = K.layers.Conv2D(
        F11, (1, 1),
        strides=s,
        kernel_initializer=initializer,
        padding='same',
        activation='linear')(A_prev)
    batch0 = K.layers.BatchNormalization()(conv0)
    activation0 = K.layers.ReLU()(batch0)
    conv1 = K.layers.Conv2D(F3, (3, 3),
                            kernel_initializer=initializer,
                            padding='same',
                            activation='linear')(activation0)
    batch1 = K.layers.BatchNormalization()(conv1)
    activation1 = K.layers.ReLU()(batch1)
    conv2 = K.layers.Conv2D(
        F12,
        (1, 1),
        kernel_initializer=initializer,
        padding='same',
        activation='linear'
    )(activation1)
    conv3 = K.layers.Conv2D(
        F12,
        (1, 1),
        strides=s,
        kernel_initializer=initializer,
        padding='same',
        activation='linear'
    )(A_prev)
    batch2 = K.layers.BatchNormalization()(conv2)
    batch3 = K.layers.BatchNormalization()(conv3)
    add = K.layers.Add()([batch2, batch3])
    return K.layers.ReLU()(add)
