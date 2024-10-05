#!/usr/bin/env python3
""" Task 1: 1. Inception Network """
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds an Inception network based on the Inception-v1 architecture.

    Returns
    model : keras.models.Model
        The Keras Model object representing the Inception network.
    """
    # implement He et. al initialization for the layers weights
    initializer = K.initializers.he_normal(seed=None)

    X = K.Input(shape=(224, 224, 3))

    # Conv 7x7 + 2(S)
    my_layer = K.layers.Conv2D(filters=64,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(X)

    # MaxPool 3x3 +n2(S)
    my_layer = K.layers.MaxPool2D(pool_size=(3, 3),
                                  padding='same',
                                  strides=(2, 2))(my_layer)

    # Conv 1x1 1(V)
    my_layer = K.layers.Conv2D(filters=64,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(my_layer)

    # Conv 3x3 1(S)
    my_layer = K.layers.Conv2D(filters=192,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(my_layer)

    # Max pooling layer with kernels of shape 3x3 with 2x2 strides
    my_layer = K.layers.MaxPool2D(pool_size=(3, 3),
                                  padding='same',
                                  strides=(2, 2))(my_layer)

    my_layer = inception_block(my_layer, [64, 96, 128, 16, 32, 32])

    my_layer = inception_block(my_layer, [128, 128, 192, 32, 96, 64])

    # Max pooling layer with kernels of shape 3x3 with 2x2 strides
    my_layer = K.layers.MaxPool2D(pool_size=(3, 3),
                                  padding='same',
                                  strides=(2, 2))(my_layer)

    my_layer = inception_block(my_layer, [192, 96, 208, 16, 48, 64])
    my_layer = inception_block(my_layer, [160, 112, 224, 24, 64, 64])
    my_layer = inception_block(my_layer, [128, 128, 256, 24, 64, 64])
    my_layer = inception_block(my_layer, [112, 144, 288, 32, 64, 64])
    my_layer = inception_block(my_layer, [256, 160, 320, 32, 128, 128])

    # Max pooling layer with kernels of shape 3x3 with 2x2 strides
    my_layer = K.layers.MaxPool2D(pool_size=(3, 3),
                                  padding='same',
                                  strides=(2, 2))(my_layer)

    my_layer = inception_block(my_layer, [256, 160, 320, 32, 128, 128])
    my_layer = inception_block(my_layer, [384, 192, 384, 48, 128, 128])

    # Avg pooling layer with kernels of shape 7x7
    my_layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same')(my_layer)

    my_layer = K.layers.Dropout(rate=0.4)(my_layer)

    # Fully connected softmax output layer with 1000 nodes
    my_layer = K.layers.Dense(units=1000,
                              activation='softmax',
                              kernel_initializer=initializer,
                              )(my_layer)

    model = K.models.Model(inputs=X, outputs=my_layer)

    return model
