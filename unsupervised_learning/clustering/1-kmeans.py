#!/usr/bin/env python3
""" Task : 1. K-means """

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Parameters:
    X : numpy.ndarray
        A 2D array of shape (n, d) containing the dataset,
        where n is the number of data points and d is the number of features.
    k : int
        The number of clusters.
    iterations : int, optional (default=1000)
        The maximum number of iterations to perform.

    Returns:
    tuple : (centroids, clss)
        centroids : numpy.ndarray
            A 2D array of shape (k, d) containing the centroid positions.
        clss : numpy.ndarray
            A 1D array of shape (n,) containing the cluster index assigned
            to each data point.
            Returns (None, None) if input validation fails.
    """
    if type(k) is not int or k <= 0:
        return None, None
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    n, d = X.shape
    centroids = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                                  size=(k, d))
    for i in range(iterations):
        copy = centroids.copy()
        D = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        clss = np.argmin(D, axis=0)
        for j in range(k):
            if len(X[clss == j]) == 0:
                centroids[j] = np.random.uniform(np.min(X, axis=0),
                                                 np.max(X, axis=0),
                                                 size=(1, d))
            else:
                centroids[j] = (X[clss == j]).mean(axis=0)
        D = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        clss = np.argmin(D, axis=0)
        if np.all(copy == centroids):
            return centroids, clss

    return centroids, clss
