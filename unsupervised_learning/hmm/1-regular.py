#!/usr/bin/env python3
""" Task 1: 1. Regular Chains """
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
    np.warnings.filterwarnings('ignore')
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

    if ((P > 0).all()):  # checks to see if all elements of P are posistive
        a = np.eye(n) - P
        a = np.vstack((a.T, np.ones(n)))
        b = np.matrix([0] * n + [1]).T
        regular = np.linalg.lstsq(a, b)[0]
        return regular.T

    return None
