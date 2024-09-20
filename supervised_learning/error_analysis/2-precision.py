#!/usr/bin/env python3
""" Task 2: 2. Precision """
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray):
        A confusion matrix of shape (classes, classes),
        where the rows represent the true labels, and the
        columns represent the predicted labels.

    Returns:
        numpy.ndarray:
        An array of shape (classes,) containing the precision
        of each class.
    """
    precision = []
    i = 0
    for row in confusion:
        positive = row[i]
        column = confusion.sum(axis=0)
        precision.append(positive / column[i])
        i = i + 1

    return np.array(precision)
