#!/usr/bin/env python3
""" Task 7: 7. Maximization """

import numpy as np


def maximization(X, g):
    """
    Performs the maximization step in the EM algorithm for a GMM.

    Parameters:
    - X (np.ndarray): Array of shape (n, d) containing the data set.
    - g (np.ndarray): Array of shape (k, n) containing the posterior
                      probabilities for each data point in each cluster.

    Returns:
    - pi (np.ndarray): Array of shape (k,) containing the updated priors
                       for each cluster.
    - m (np.ndarray): Array of shape (k, d) containing the updated
                      centroid means for each cluster.
    - S (np.ndarray): Array of shape (k, d, d) containing the updated
                      covariance matrices for each cluster.
                      Returns (None, None, None) on failure.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None
    cluster = np.sum(g, axis=0)
    cluster = np.sum(cluster)
    if int(cluster) != X.shape[0]:
        return None, None, None

    n, d = X.shape
    k, n = g.shape

    # nk is the sum of posterior probabilities
    nk = np.sum(g, axis=1)
    # The task here is update priors(pi), mean and covariance(cov)
    # pi (also call weights) is nk / the total number of points
    pi = nk / n
    mean = np.zeros((k, d))
    cov = np.zeros((k, d, d))
    for i in range(k):
        mean[i] = np.matmul(g[i], X) / nk[i]
        norm = X - mean[i]
        cov[i] = np.matmul(g[i] * norm.T, norm) / nk[i]

    return pi, mean, cov
