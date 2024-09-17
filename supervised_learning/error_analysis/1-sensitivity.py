#!/usr/bin/env python3
""" Task 1: 1. Sensitivity """
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (true positive rate or recall) for each class
    from the confusion matrix.

    Sensitivity is calculated as the ratio of true positives to the
    sum of true positives and false negatives:
    Sensitivity = TP / (TP + FN)

    Parameters:
    confusion : numpy.ndarray
        A square confusion matrix of shape (n, n),
        where `n` is the number of classes.
        The element at [i, j] represents the number of
        times class `i` was predicted as class `j`.

    Returns:
    numpy.ndarray
        An array of shape (n,) containing the sensitivity for each class.

    """
    sensitivity = []
    i = 0
    for row in confusion:
        positive = row[i]
        sensitivity.append(positive / false_positive)
        i = i + 1

    return np.array(sensitivity)
