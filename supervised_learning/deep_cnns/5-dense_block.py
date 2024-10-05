#!/usr/bin/env python3
""" Task 5: 5. Dense Block  """
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block for the DenseNet architecture.

    Parameters
    X : tensor
        Input tensor of shape (height, width, channels).
    nb_filters : int
        The number of filters before entering the dense block.
    growth_rate : int
        The number of filters to be added after each convolution (growth rate).
    layers : int
        The number of convolutional layers within the dense block.

    Returns
    X : tensor
        Output tensor after passing through the dense block.
    nb_filters : int
        The number of filters after the dense block.
    """
    # implement He et. al initialization for the layers weights
    initializer = K.initializers.he_normal(seed=0)

    for i in range(layers):
        my_layer = K.layers.BatchNormalization()(X)
        my_layer = K.layers.Activation(activation='relu')(my_layer)

        # conv 1×1 produces 4k (growth_rate) feature-maps
        my_layer = K.layers.Conv2D(filters=4*growth_rate,
                                   kernel_size=1,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   )(my_layer)

        my_layer = K.layers.BatchNormalization()(my_layer)
        my_layer = K.layers.Activation('relu')(my_layer)

        # conv 3×3 produces k (growth_rate) feature-maps
        my_layer = K.layers.Conv2D(filters=growth_rate,
                                   kernel_size=3,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   )(my_layer)

        X = K.layers.concatenate([X, my_layer])
        nb_filters += growth_rate

    return X, nb_filters
