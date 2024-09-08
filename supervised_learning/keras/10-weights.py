#!/usr/bin/env python3
""" Task 10: 10. Save and Load Weights """
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Saves a model’s weights to a file.

    Args:
        network (keras.Model): The model whose weights should be saved.
        filename (str): Path to the file where the weights should be saved.
        save_format (str, optional): Format in which the weights should be
                                    saved (default is 'h5').

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Loads a model’s weights from a file.

    Args:
        network (keras.Model): The model to which the weights should be loaded.
        filename (str): Path to the file from which the weights should
                        be loaded.

    Returns:
        None
    """
    network.load_weights(filename)
    return None
