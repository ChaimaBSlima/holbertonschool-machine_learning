#!/usr/bin/env python3
""" Task 4: 4. Initialize GMM """
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes parameters for a Gaussian Mixture Model (GMM).

    Parameters:
    X : numpy.ndarray
        A 2D array of shape (n, d) containing the dataset,
        where n is the number of data points and d is the number of features.
    k : int
        The number of clusters.

    Returns:
    tuple : (pi, m, S)
        pi : numpy.ndarray
            A 1D array of shape (k,) containing the initialized priors,
            where each cluster is given equal weight.
        m : numpy.ndarray
            A 2D array of shape (k, d) containing the initialized centroids
            from K-means clustering.
        S : numpy.ndarray
            A 3D array of shape (k, d, d) containing the initialized
            covariance matrices, initialized as identity matrices.
        Returns (None, None, None) if input validation fails.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None

    n, d = X.shape
    centroids, clss = kmeans(X, k, iterations=1000)
    pi = np.ones(k) / k
    m = centroids
    S = np.tile(np.identity(d), (k, 1)).reshape((k, d, d))

    return pi, m, S
