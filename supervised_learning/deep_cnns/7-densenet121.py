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

    initializer = K.initializers.HeNormal(seed=0)

    X = K.Input(shape=(224, 224, 3))
    layers = [12, 24, 16]

    # Initial Batch Normalization and ReLU
    my_layer = K.layers.BatchNormalization(axis=-1)(X)
    my_layer = K.layers.Activation.ReLU()(my_layer)

    # Convolution 7x7 + Strides 2
    my_layer = K.layers.Conv2D(filters=2 * growth_rate,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=initializer)(my_layer)

    # MaxPool 3x3 + Strides 2
    my_layer = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     padding='same',
                                     strides=(2, 2))(my_layer)

    nb_filters = 2 * growth_rate

    # Dense block
    my_layer, nb_filters = dense_block(my_layer, nb_filters, growth_rate, 6)

    for layer in layers:
        # Transition layer
        my_layer, nb_filters = transition_layer(my_layer,
                                                nb_filters,
                                                compression)

        # Dense block
        my_layer, nb_filters = dense_block(my_layer,
                                           nb_filters,
                                           growth_rate,
                                           layer)

    # Classification layer
    # Average pooling layer with kernels of shape 7x7
    my_layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same')(my_layer)

    # Fully connected softmax output layer with 1000 nodes
    my_layer = K.layers.Dense(units=1000,
                              activation='softmax',
                              kernel_initializer=initializer)(my_layer)

    model = K.models.Model(inputs=X, outputs=my_layer)

    return model
