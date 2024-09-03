#!/usr/bin/env python3
""" Task 2: 2. Forward Propagation """
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for the neural network

    Args:
        x:  placeholder for the input data
        layer_sizes: list containing the number of nodes in
            each layer of the network
        activations: list containing the activation functions
            for each layer of the network
    
    Returns:
        The prediction of the network in tensor form.
    """
    layer = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer
