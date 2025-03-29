#!/usr/bin/env python3
""" Task 0: 0. Mean and Covariance """

import numpy as np


def mean_cov(X):
    """
    Computes the mean vector and covariance matrix of a given dataset.

    Parameters:
    X (numpy.ndarray): A 2D numpy array of shape (n, d), where:
                       - n is the number of data points.
                       - d is the number of features.

    Returns:
    tuple: A tuple (mean, cov) where:
    - mean (numpy.ndarray): A 1D array of shape (1, d)
    representing the mean of each feature.
    - cov (numpy.ndarray): A 2D array of shape (d, d)

    Raises:
    TypeError: If X is not a 2D numpy array.
    ValueError: If X contains fewer than two data points.

    Notes:
    - The mean is computed as the average of all data points along
        each feature.
    - The covariance matrix is computed using the unbiased estimator:
      cov = (1 / (n-1)) * Σ [(xi - mean) @ (xi - mean)^T]
    - This function follows the standard statistical definitions for
    mean and covariance.

    Reference:
    - https://en.wikipedia.org/wiki/Sample_mean_and_covariance
    """

    if (isinstance(X, type(None))):
        raise TypeError('X must be a 2D numpy.ndarray')

    if (not isinstance(X, np.ndarray)) or (len(X.shape) != 2):
        raise TypeError('X must be a 2D numpy.ndarray')

    if (X.shape[0] < 2):
        raise ValueError("X must contain multiple data points")

    # Sample mean vector =  [1 / n] *  ⅀ [(xi  - x̅) * (xi  - x̅)ₜ]
    mean = X.mean(axis=0)
    mean = np.reshape(mean, (-1, X.shape[1]))

    # Sample cov matrix. It is a matrix Q = q sub(ij) of size d x d where
    # Q = [1 / (n-1)] * ⅀ [(xi  - x̅) * (xi  - x̅)^T]
    n = X.shape[0] - 1
    x = X - mean

    cov = np.dot(x.T, x) / n

    return mean, cov
