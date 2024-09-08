#!/usr/bin/env python3
""" Task 12: 12. Test """

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network.

    Args:
        network (keras.Model): The network model to test.
        data (numpy.ndarray): Input data to test the model with.
        labels (numpy.ndarray): Correct one-hot encoded labels for the data.
        verbose (bool): Whether to print output during the testing process.

    Returns:
        tuple: Loss and accuracy of the model on the test data, respectively.
    """
    return network.evaluate(data, labels, verbose=verbose)
