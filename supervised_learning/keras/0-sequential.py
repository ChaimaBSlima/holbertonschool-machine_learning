#!/usr/bin/env python3
""" Task 0: 0. Sequential """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a fully connected neural network using the Keras library.

    Parameters:

    nx : int
        The number of input features to the network.
    layers : list of int
        A list containing the number of nodes in each layer
        of the network.
    activations : list of str or callable
        A list containing the activation functions for
        each layer.
    lambtha : float
        The L2 regularization parameter (L2 penalty) for
        kernel weights.
    keep_prob : float
        The probability of keeping a node active during dropout
        (1 - dropout rate).

    Returns:
    model : keras.Sequential
        The constructed Keras model.
    """
    model = K.Sequential()

    reg = K.regularizers.L1L2(l2=lambtha)

    model.add(K.layers.Dense(units=layers[0],
                             activation=activations[0],
                             kernel_regularizer=reg,
                             input_shape=(nx,),))

    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(units=layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=reg,
                                 ))

    return model
