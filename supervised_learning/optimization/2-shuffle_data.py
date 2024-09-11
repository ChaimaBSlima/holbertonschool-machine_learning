#!/usr/bin/env python3
""" Task 2: 2. Shuffle Data """
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices in the same way,
    ensuring corresponding rows remain aligned.

    Args:
        X (numpy.ndarray): A 2D array of shape (m, nx) where:
            - m is the number of data points.
            - nx is the number of features in X.
        Y (numpy.ndarray): A 2D array of shape (m, ny) where:
            - m is the same number of data points as in X.
            - ny is the number of features in Y.

    Returns:
        tuple: A tuple of two numpy arrays (X_shuffled, Y_shuffled), where:
            - X_shuffled is the shuffled version of X.
            - Y_shuffled is the shuffled version of Y, shuffled in the
                order as X.
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    return X[shuffle], Y[shuffle]
