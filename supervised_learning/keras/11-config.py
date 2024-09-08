#!/usr/bin/env python3
""" Task 11: 11. Save and Load Configuration """

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format.

    Args:
        network (keras.Model): The model whose configuration should be saved.
        filename (str): Path to the file where the configuration will be saved.

    Returns:
        None
    """
    json_string = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_string)
    return None


def load_config(filename):
    """
    Loads a model’s configuration from a JSON file.

    Args:
        filename (str): Path to the file containing the model’s
        configuration in JSON format.

    Returns:
        keras.Model: The model created from the loaded configuration.
    """
    with open(filename, "r") as f:
        network_string = f.read()
    network = K.models.model_from_json(network_string)
    return network
