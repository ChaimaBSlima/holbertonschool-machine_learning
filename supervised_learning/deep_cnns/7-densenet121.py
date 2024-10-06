#!/usr/bin/env python3
""" Task 7: 7. DenseNet-121 """
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture.

    Parameters
    growth_rate : int, optional
        The growth rate, which determines the number of filters to
        be added for each dense block.
        Default is 32.
    compression : float, optional
        The compression factor to reduce the number of filters in the
        transition layers.
        Default is 1.0 (no compression).

    Returns
    model : keras.Model
        A Keras Model instance representing the DenseNet-121 architecture.

    """

    he_normal = K.initializers.he_normal(seed=0)  # Initialize with He normal and seed 0

    # Input layer
    input = layers.Input(shape=(224, 224, 3))

    # Initial Conv layer (7x7, stride 2)
    x = K.layers.BatchNormalization()(input)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', kernel_initializer=he_normal)(x)

    # MaxPooling layer (3x3, stride 2)
    x = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Dense blocks with transition layers
    x,num_filters = dense_block(x, 6, growth_rate)  # First dense block (6 layers)
    x = transition_layer(x, int(num_filters * compression))  # First transition layer

    x, num_filters = dense_block(x, 12, growth_rate)  # Second dense block (12 layers)
    x = transition_layer(x, int(num_filters * compression))  # Second transition layer

    x, num_filters = dense_block(x, 24, growth_rate)  # Third dense block (24 layers)
    x = transition_layer(x, int(num_filters * compression))  # Third transition layer

    x, num_filters = dense_block(x, 16, growth_rate)  # Fourth dense block (16 layers)

    # Classification layer
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.GlobalAveragePooling2D()(x)
    output = K.layers.Dense(1000, activation='softmax', kernel_initializer=he_normal)(x)

    # Model creation
    model = K.models.Model(inputs=input, outputs=output)

    return model