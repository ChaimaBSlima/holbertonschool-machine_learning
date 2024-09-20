#!/usr/bin/env python3
""" Task 3: 3. Specificity """
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Specificity is also known as the true negative rate, and it measures
    the proportion of actual negatives that are correctly identified as such.

    Args:
        confusion (numpy.ndarray):
        A confusion matrix of shape (classes, classes),
        where the rows represent the true labels, and the
        columns represent the predicted labels.

    Returns:
        numpy.ndarray:
        An array of shape (classes,) containing the specificity
        of each class.
    """
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)

    return (TN / (TN + FP))
