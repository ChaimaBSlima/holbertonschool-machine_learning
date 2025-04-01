#!/usr/bin/env python3
""" Task 3: 3. Optimize k """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Determines the optimum number of clusters using the
    variance difference method.

    Parameters:
    X : numpy.ndarray
        A 2D array of shape (n, d) containing the dataset,
        where n is the number of data points and d is the number of features.
    kmin : int, optional (default=1)
        The minimum number of clusters to check.
    kmax : int, optional (default=None)
        The maximum number of clusters to check.
        If None, it defaults to the number of data points.
    iterations : int, optional (default=1000)
        The maximum number of iterations for the K-means algorithm.

    Returns:
    tuple : (results, d_vars)
        results : list of tuples
            Each tuple contains (centroids, cluster assignments) for a given k.
        d_vars : list of floats
            The differences in variance as k increases.
        Returns (None, None) if input validation fails.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    if type(kmin) is not int or kmin < 1:
        return None, None
    if kmax is not None and (type(kmax) is not int or kmax < 1):
        return None, None
    if kmax is not None and kmin >= kmax:
        return None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    results = []
    d_vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations=1000)
        results.append((C, clss))

        if k == kmin:
            first_var = variance(X, C)
        var = variance(X, C)
        d_vars.append(first_var - var)

    return results, d_vars
