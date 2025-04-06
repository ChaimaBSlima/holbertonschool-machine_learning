#!/usr/bin/env python3
""" Task 9: 9. BIC """
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion (BIC).

    Parameters:
    - X (np.ndarray): Array of shape (n, d) containing the data set.
    - kmin (int): Minimum number of clusters to check for (inclusive).
    - kmax (int): Maximum number of clusters to check for (inclusive).
                  If None, set to number of data points.
    - iterations (int): Maximum number of iterations for the EM algorithm.
    - tol (float): Tolerance for the EM algorithm's log likelihood convergence.
    - verbose (bool): If True, prints information during EM execution.

    Returns:
    - best_k (int): The best number of clusters based on BIC.
    - best_result (tuple): Tuple containing (pi, m, S) for the
        best number of clusters.
    - l (np.ndarray): Log likelihoods for each number of clusters tested.
    - b (np.ndarray): BIC values for each number of clusters tested.
    Returns (None, None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmin) is not int or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None
    if type(kmax) is not int or kmax <= 0 or X.shape[0] < kmax:
        return None, None, None, None
    if kmax <= kmin:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    n, d = X.shape

    b = []
    results = []
    ks = []
    l_ = []

    for k in range(kmin, kmax + 1):
        ks.append(k)

        pi, m, S, g, l_k = expectation_maximization(X,
                                                    k,
                                                    iterations=iterations,
                                                    tol=tol,
                                                    verbose=verbose)
        results.append((pi, m, S))

        l_.append(l_k)
        p = k - 1 + k * d + k * d * (d + 1) / 2

        bic = p * np.log(n) - 2 * l_k
        b.append(bic)

    l_ = np.array(l_)
    b = np.array(b)

    index = np.argmin(b)
    best_k = ks[index]
    best_result = results[index]

    return best_k, best_result, l_, b
