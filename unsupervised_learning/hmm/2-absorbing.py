#!/usr/bin/env python3
""" Task 2: 2. Absorbing Chains """
import numpy as np


def markov_chain(P, s, t=1):
    """
    Performs t iterations of a Markov chain given a transition matrix P and a
    starting state s.

    Parameters:
    - P: 2D numpy.ndarray of shape (n, n) representing the transition matrix
    - s: 2D numpy.ndarray of shape (1, n) representing the probability of
      starting in each state
    - t: Number of iterations (positive integer)

    Returns:
    - A 2D numpy.ndarray of shape (1, n) representing the probability of
      being in a particular state after t iterations, or None on failure.
    """
    if (not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray)):
        return None

    if (not isinstance(t, int)):
        return None

    if ((P.ndim != 2) or (s.ndim != 2) or (t < 1)):
        return None

    n = P.shape[0]
    if (P.shape != (n, n)) or (s.shape != (1, n)):
        return None

    while (t > 0):
        s = np.matmul(s, P)
        t -= 1

    return s


def regular(P):
    """
    Determines the steady state of a regular Markov chain.

    Parameters:
    - P: 2D numpy.ndarray of shape (n, n) representing the transition matrix

    Returns:
    - A 1D numpy.ndarray of shape (1, n) representing the steady state,
      or None if not regular or input is invalid.
    """
    # import warning
    # warnings.filterwarnings('ignore')
    # Avoid this warning: Line 92.  np.linalg.lstsq(a, b)[0]

    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None

    n = P.shape[0]
    if P.shape != (n, n):
        return None

    if not np.allclose(P.sum(axis=1), 1):
        return None

    if (P > 0).all():
        a = np.eye(n) - P
        a = np.vstack((a.T, np.ones(n)))
        b = np.array([0] * n + [1])
        regular = np.linalg.lstsq(a, b, rcond=None)[0]
        return regular[np.newaxis, :]

    return None


def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    Parameters:
    - P (numpy.ndarray): A square 2D transition matrix.

    Returns:
    - True if the Markov chain is absorbing, False if not,
        or None if input is invalid.
    """
    # import warning
    # warnings.filterwarnings('ignore')
    # Avoid this warning: Line 92.  np.linalg.lstsq(a, b)[0]

    if (not isinstance(P, np.ndarray)):
        return None

    if (P.ndim != 2):
        return None

    n = P.shape[0]
    if (P.shape != (n, n)):
        return None

    if ((np.sum(P) / n) != 1):
        return None

    # P is an identity matrix
    identity = np.eye(n)
    if (np.equal(P, identity).all()):
        return True

    abs = np.zeros(n)
    for i in range(n):
        if P[i][i] == 1:
            abs[i] = 1

    prev = np.zeros(n)
    while (not np.array_equal(abs, prev) and abs.sum() != n):
        prev = abs.copy()
        for absorbed in P[:, np.nonzero(abs)[0]].T:
            abs[np.nonzero(absorbed)] = 1
    if (abs.sum() == n):
        return True
    return False
