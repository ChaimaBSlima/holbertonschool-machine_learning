#!/usr/bin/env python3
""" Task 2: 2. Variance """

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a given
    set of cluster centroids.

    Parameters:
    X : numpy.ndarray
        A 2D array of shape (n, d) containing the dataset,
        where n is the number of data points and d is the
        number of features.
    C : numpy.ndarray
        A 2D array of shape (k, d) containing the cluster
        centroids, where k is the number of clusters.

    Returns:
    float
        The total variance, calculated as the sum of squared distances
        of each data point to its closest centroid.
        Returns None if input validation fails.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    k, d = C.shape
    if type(k) is not int or k <= 0:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    D = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=2))
    cluster = np.min(D, axis=0)

    var = np.sum(np.square(cluster))
    return var
