#!/usr/bin/env python3
""" Task 0: 0. Normalization Constants """
import numpy as np


def normalization_constants(X):
    """
     Calculates the mean and standard deviation of a given dataset X.

    Args:
        X (numpy.ndarray): A 2D array where each column represents
        a feature and each row represents a data point.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - m (numpy.ndarray): The mean of each feature
                                (shape: (n_features,))
            - s (numpy.ndarray): The standard deviation of each feature
                                (shape: (n_features,))
    """
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s
