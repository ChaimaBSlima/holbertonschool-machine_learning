#!/usr/bin/env python3
""" Task 12: 12. Agglomerative """
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative hierarchical clustering on the dataset.

    Parameters:
    - X (np.ndarray): The dataset of shape (n, d).
    - dist (float): The distance threshold for forming clusters.

    Returns:
    - fcluster (np.ndarray): The cluster labels for each
    data point based on the distance threshold.
    
    Saves:
    - A dendrogram plot as "12-figFile.png" that visualizes the
    hierarchical clustering.
    """
    hierarchy = scipy.cluster.hierarchy
    linkage = hierarchy.linkage(X, method='ward')
    fcluster = hierarchy.fcluster(linkage, dist, criterion='distance')

    hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.figure()
    plt.savefig("12-figFile")

    return fcluster
