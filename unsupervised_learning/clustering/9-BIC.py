#!/usr/bin/env python3
""" clustering """
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Function that performs the expectation maximization for a GMM:
    Args:
                X: numpy.ndarray -> Array of shape (n, d) with the data
        kmin: int -> positive int with the minimum number
            of clusters to check for (inclusive)
        kmax: int -> positive int with the maximum number
            of clusters to check for (inclusive)
        iterations: int -> positive int with the maximum number
            of iterations for the algorithm
        tol: float -> non-negative float with the tolerance
            of the log likelihood, used to
            determine early stopping i.e. if the
            difference is less than or equal to
            tol you should stop the algorithm
        verbose: bool -> boolean that determines if you should
            print information about the algorithm
            If True, print Log Likelihood after {i} iterations: {l}
            every 10 iterations and after the last iteration
                {i} is the number of iterations of the EM algorithm
                 {l} is the log likelihood
            You should use:
            initialize = __import__('4-initialize').initialize
            expectation = __import__('6-expectation').expectation
            maximization = __import__('7-maximization').maximization
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmin) != int or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None
    if type(kmax) != int or kmax <= 0 or X.shape[0] < kmax:
        return None, None, None, None
    if kmax <= kmin:
        return None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None
    if type(verbose) != bool:
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
