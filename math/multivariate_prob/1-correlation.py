#!/usr/bin/env python3
""" Task 1: 1. Correlation """

import numpy as np


def correlation(C):
    """
    Computes the correlation matrix from a given covariance matrix.

    Parameters:
    C (numpy.ndarray):
    A 2D square numpy array of shape (d, d), where:
     - d is the number of features.
    - C represents the covariance matrix.

    Returns:
    numpy.ndarray: A 2D array of shape (d, d)
    representing the correlation matrix.

    Raises:
    TypeError: If C is not a numpy array.
    ValueError: If C is not a 2D square matrix.

    Notes:
    - The correlation matrix is computed using the formula:
      correlation[i, j] = C[i, j] / (sqrt(C[i, i]) * sqrt(C[j, j]))
    - The diagonal of the correlation matrix will always be 1.
    - This function ensures that the input matrix is a valid
        covariance matrix.

    Reference:
    - https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    """
    if (isinstance(C, type(None))):
        raise TypeError('C must be a numpy.ndarray')

    if (not isinstance(C, np.ndarray)):
        raise TypeError('C must be a numpy.ndarray')

    if (len(C.shape) != 2):
        raise ValueError("C must be a 2D square matrix")

    if (C.shape[0] != C.shape[1]):
        raise ValueError("C must be a 2D square matrix")

    v = np.sqrt(np.diag(C))
    outer_v = np.outer(v, v)
    correlation = C / outer_v

    return correlation
