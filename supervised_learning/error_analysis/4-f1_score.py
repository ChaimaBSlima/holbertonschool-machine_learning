#!/usr/bin/env python3
""" Task 4: 4. F1 score """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
     Calculates the F1 score for each class in a confusion matrix.

    The F1 score is the harmonic mean of precision and sensitivity
    (also known as recall). It provides a balanced measure of precision
    and recall, especially useful in cases where the class distribution
    is imbalanced.

    Args:
        confusion (numpy.ndarray):
        A confusion matrix of shape (classes, classes),
        where the rows represent the true labels and the
        columns represent the predicted labels.

    Returns:
        numpy.ndarray:
        An array of shape (classes,) containing the F1 score for
            each class.
    """
    prec = precision(confusion)
    sens = sensitivity(confusion)
    return 2 * (prec * sens)/(prec + sens)
    TPR = TP / (TP + FN)
    PPV = TP / (TP + FP)
    F1 = 2 / ((1 / TPR) + (1 / PPV))
    return F1
