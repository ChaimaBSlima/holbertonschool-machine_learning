#!/usr/bin/env python3
""" Task 11: 11. GMM """
import sklearn.mixture


def gmm(X, k):
    """
    Performs Gaussian Mixture Model (GMM) clustering on the dataset.

    Parameters:
    - X (np.ndarray): The dataset of shape (n, d).
    - k (int): The number of clusters (components) for the GMM.

    Returns:
    - pi (np.ndarray): The weights (priors) for each cluster.
    - m (np.ndarray): The means (centroids) for each cluster, shape (k, d).
    - S (np.ndarray): The covariances for each cluster, shape (k, d, d).
    - clss (np.ndarray): The predicted cluster label for each data point, shape (n,).
    - bic (float): The Bayesian Information Criterion (BIC) for the model.
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
