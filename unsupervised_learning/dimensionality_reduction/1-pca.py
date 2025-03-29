#!/usr/bin/env python3
"""Task 1: 1. PCA v2"""
import numpy as np


def pca(X, ndim):
    """
    Performs Principal Component Analysis (PCA) to reduce the
    dimensionality of a dataset.

    Parameters:
    X (numpy.ndarray): A 2D numpy array of shape (n, d), where:
                       - n is the number of data points.
                       - d is the number of original features.
    ndim (int): The number of principal components to retain.

    Returns:
    numpy.ndarray: A transformed dataset of shape (n, ndim), where:
                   - ndim is the reduced number of features.

    Notes:
    - The input data is centered by subtracting the mean of each feature.
    - Singular Value Decomposition (SVD) is used to compute the principal
        components.
    - The transformation matrix W consists of the first `ndim` principal
        components from the right singular vector matrix (Vh).
    - The transformed dataset is obtained by projecting X onto the
        principal components.

    Reference:
    - https://en.wikipedia.org/wiki/Principal_component_analysis
    """
    X_m = X - np.mean(X, axis=0)

    u, s, vh = np.linalg.svd(X_m)

    W = vh[:ndim].T
    T = np.matmul(X_m, W)

    return T
