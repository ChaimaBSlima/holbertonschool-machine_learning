#!/usr/bin/env python3
""" Task 6: 6. Expectation """

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the expectation step in the EM algorithm for a GMM.

    Parameters:
    - X (np.ndarray): Array of shape (n, d)
        containing the data set.
    - pi (np.ndarray): Array of shape (k,)
        containing the priors for each cluster.
    - m (np.ndarray): Array of shape (k, d)
        containing the centroid means for each cluster.
    - S (np.ndarray): Array of shape (k, d, d)
        containing the covariance matrices for each cluster.

    Returns:
    - g (np.ndarray):
    Array of shape (k, n) containing the posterior probabilities
    for each data point in each cluster.
    - log_likelihood (float): Total log likelihood for the data
        given the current parameters.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None
    k = pi.shape[0]
    n, d = X.shape

    if k > n:
        return None, None
    if d != m.shape[1] or d != S.shape[1] or d != S.shape[2]:
        return None, None
    if k != m.shape[0] or k != S.shape[0]:
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    probs = np.zeros((k, n))
    for i in range(k):
        probs[i] = pi[i] * pdf(X, m[i], S[i])

    marginal = np.sum(probs, axis=0)
    g = probs / marginal
    log_likelihood = np.sum(np.log(marginal))

    return g, log_likelihood
