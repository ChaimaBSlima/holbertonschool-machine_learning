#!/usr/bin/env python3
""" Task 8: 8. EM """

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the Expectation-Maximization (EM) algorithm for a
    Gaussian Mixture Model (GMM).

    Parameters:
    - X (np.ndarray): Array of shape (n, d) containing the data set.
    - k (int): Number of clusters.
    - iterations (int): Maximum number of iterations for the algorithm.
    - tol (float): Tolerance for the log likelihood change to declare
        convergence.
    - verbose (bool): If True, prints the log likelihood every 10
        iterations and at the end.

    Returns:
    - pi (np.ndarray): Array of shape (k,) containing the priors for
        each cluster.
    - m (np.ndarray): Array of shape (k, d) containing the centroid
        means for each cluster.
    - S (np.ndarray): Array of shape (k, d, d) containing the
        covariance matrices for each cluster.
    - g (np.ndarray): Array of shape (k, n) containing the posterior
        probabilities for each data point.
    - log_likelihood (float): The final log likelihood.
    Returns (None, None, None, None, None) on failure.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    i = 0
    l_prev = 0
    pi, mean, cov = initialize(X, k)
    g, log_like = expectation(X, pi, mean, cov)
    while i < iterations:
        if (np.abs(l_prev - log_like)) <= tol:
            break
        l_prev = log_like

        if verbose is True and (i % 10 == 0):
            rounded = log_like.round(5)
            print("Log Likelihood after {} iterations: {}".format(i, rounded))

        pi, mean, cov = maximization(X, g)
        g, log_like = expectation(X, pi, mean, cov)
        i += 1

    if verbose is True:
        rounded = log_like.round(5)
        print("Log Likelihood after {} iterations: {}".format(i, rounded))

    return pi, mean, cov, g, log_like
