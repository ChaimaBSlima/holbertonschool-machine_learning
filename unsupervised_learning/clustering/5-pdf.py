#!/usr/bin/env python3
""" Task 5: 5. PDF """
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a multivariate 
    normal distribution.

    Parameters:
    - X (np.ndarray): A matrix of shape (n, d) containing the data points
                      whose PDF should be evaluated.
    - m (np.ndarray): A numpy array of shape (d,) containing the mean
                      of the distribution.
    - S (np.ndarray): A numpy array of shape (d, d) containing the
                      covariance matrix of the distribution.

    Returns:
    - P (np.ndarray): A numpy array of shape (n,) containing the PDF 
                      values for each data point, or None on failure.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None
    d = S.shape[0]

    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    first = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))
    second = np.dot((X - m), inv)
    third = np.sum(second * (X - m) / -2, axis=1)
    P = first * np.exp(third)

    P = np.maximum(P, 1e-300)

    return P
