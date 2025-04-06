#!/usr/bin/env python3
""" Task 10: 10. Hello, sklearn! """
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means clustering on a dataset.

    Parameters:
    - X (np.ndarray): The dataset of shape (n, d).
    - k (int): The number of clusters.

    Returns:
    - C (np.ndarray): Centroid coordinates for each
    cluster, shape (k, d).
    - clss (np.ndarray): Index of the cluster each data
    point belongs to, shape (n,).
    """
    k_means = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = k_means.cluster_centers_
    clss = k_means.labels_
    return C, clss
