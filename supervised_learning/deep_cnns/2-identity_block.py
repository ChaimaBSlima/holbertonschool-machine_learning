#!/usr/bin/env python3
""" Task 2: 2. Identity Block """
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds a projection block as part of a residual network.
    """
    F11, F3, F12 = filters

    # implement He et. al initialization for the layers weights
    initializer = K.initializers.he_normal(seed=None)

    # Conv 1x1
    my_layer = K.layers.Conv2D(filters=F11,
                               kernel_size=(1, 1),
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

    output = K.layers.Add()([my_layer, A_prev])

    output = K.layers.Activation('relu')(output)

    return output
