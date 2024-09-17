#!/usr/bin/env python3
""" Task 0: 0. Create Confusion """
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix by computing the dot
    product of the labels and logits.

    Parameters:
    labels : numpy.ndarray
        A numpy array of shape (m, n), where `m` is the number of samples and
        `n` is the number of classes.
        It contains the true class labels in a one-hot encoded format.

    logits : numpy.ndarray
        A numpy array of shape (m, n), where `m` is the number of samples
        and `n` is the number of classes.
        It contains the predicted class probabilities or outputs, typically
        as the result of a softmax or sigmoid function.

    Returns:
    numpy.ndarray
        The confusion matrix of shape (n, n), where `n` is
        the number of classes.
        Each element at (i, j) represents the number of times
        class `i` was predicted as class `j`.

    """
    return np.dot(labels.T, logits)
