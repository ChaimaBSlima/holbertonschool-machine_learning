#!/usr/bin/env python3
""" Task 3: 3. One Hot """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot encoded
    matrix using Keras utilities.

    Parameters:
    -----------
    labels : array-like
        A vector of labels to be converted into one-hot encoding.
    classes : int, optional
        The total number of classes. If not provided,
        it will be inferred from the data.

    Returns:
    --------
    one_hot_matrix : numpy.ndarray
        A one-hot encoded matrix representing
        the input labels.
    """
    return K.utils.to_categorical(
        labels, classes)
