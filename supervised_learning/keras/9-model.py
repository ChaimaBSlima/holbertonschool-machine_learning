#!/usr/bin/env python3
""" Task 9: 9. Save and Load Model """
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model to a file.

    Args:
        network (keras.Model): The model to save.
        filename (str): Path to the file where the model should be saved.

    Returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model from a file.

    Args:
        filename (str): Path to the file from which the model should be loaded.

    Returns:
        keras.Model: The loaded model.
    """
    return K.models.load_model(filename)
