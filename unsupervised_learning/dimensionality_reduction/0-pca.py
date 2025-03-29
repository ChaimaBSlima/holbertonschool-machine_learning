#!/usr/bin/env python3
""" Task 0: 0. PCA"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs Principal Component Analysis (PCA) on the given dataset.

    Parameters:
    X (numpy.ndarray): A 2D numpy array of shape (n, d), where:
                       - n is the number of data points.
                       - d is the number of features.
    var (float, optional):
    The desired explained variance ratio. Default is 0.95.

    Returns:
    numpy.ndarray: A transformation matrix of shape (d, k), where:
    - k is the number of principal components needed to preserve
    the given variance.

    Notes:
    - The function uses Singular Value Decomposition (SVD) to compute PCA.
    - The cumulative sum of singular values is used to determine the
        number of components (k) required to retain the desired variance.
    - The transformation matrix consists of the first k principal components
        from the right singular vector matrix (Vh from SVD).

    Reference:
    - https://en.wikipedia.org/wiki/Principal_component_analysis
    """
    u, s, vh = np.linalg.svd(X)

    cumsum = np.cumsum(s)

    dim = [i for i in range(len(s)) if cumsum[i] / cumsum[-1] >= var]
    ndim = dim[0] + 1

    return vh.T[:, :ndim]
