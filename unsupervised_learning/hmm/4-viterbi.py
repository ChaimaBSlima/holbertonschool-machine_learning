#!/usr/bin/env python3
""" Task 4: 4. The Viretbi Algorithm """
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
    # imp_ort warning
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
    # impo_rt warning
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


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.

    Parameters:
    - Observation (numpy.ndarray): shape (T,) with the index of observations.
    - Emission (numpy.ndarray): shape (N, M), emission probabilities.
    - Transition (numpy.ndarray): shape (N, N), transition probabilities.
    - Initial (numpy.ndarray): shape (N, 1), initial state probabilities.

    Returns:
    - likelihood (float): total probability of the observed sequence.
    - forward (numpy.ndarray): shape (N, T), forward path probabilities.
    """
    try:
        if (not isinstance(Observation, np.ndarray)) or (
                not isinstance(Emission, np.ndarray)) or (
                not isinstance(Transition, np.ndarray)) or (
                not isinstance(Initial, np.ndarray)):
            return None, None

        # 2. Dim validations
        if (Observation.ndim != 1) or (
                Emission.ndim != 2) or (
                Transition.ndim != 2) or (
                Initial.ndim != 2):
            return None, None

        # 3. Structure validations
        if (not np.sum(Emission, axis=1).all() == 1) or (
                not np.sum(Transition, axis=1).all() == 1) or (
                not np.sum(Initial).all() == 1):
            return None, None

        T = Observation.shape[0]
        N = Emission.shape[0]

        forward = np.zeros((N, T))
        forward[:, 0] = Initial.T * Emission[:, Observation[0]]

        for t in range(1, T):
            for j in range(N):
                forward[j, t] = (forward[:, t - 1].dot(Transition[:, j])
                                 * Emission[j, Observation[t]])

        likelihood = np.sum(forward[:, t])
        return likelihood, forward

    except BaseException:
        return None, None


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for
    a hidden Markov model.

    Parameters:
    - Observation (np.ndarray):
        shape (T,), index of each observation
    - Emission (np.ndarray):
        shape (N, M), emission probability of a specific observation
    - Transition (np.ndarray):
        shape (N, N), transition probabilities between states
    - Initial (np.ndarray):
        shape (N, 1), initial state probabilities

    Returns:
    - path (list[int]): the most likely sequence of hidden states
    - probability (float): probability of obtaining that sequence
    """
    N = Emission.shape[0]

    try:
        if (Observation.ndim != 1 or Emission.ndim != 2):
            return None, None

        if (Transition.shape != (N, N) or Initial.shape != (N, 1)):
            return None, None

        if (not np.isclose(np.sum(Emission, axis=1), 1).all()):
            return None, None

        if (not np.isclose(np.sum(Transition, axis=1), 1).all()):
            return None, None

        if (not np.isclose(np.sum(Initial, axis=0), 1).all()):
            return None, None

        T = Observation.shape[0]
        F = np.zeros((N, T))
        prev = np.zeros((N, T))

        # Initilaize the tracking tables from first observation
        F[:, 0] = Initial.T * Emission[:, Observation[0]]
        prev[:, 0] = 0

        # Iterate throught the observations updating the tracking tables
        for i, obs in enumerate(Observation):
            if i != 0:
                F[:, i] = np.max(F[:, i - 1] * Transition.T *
                                 Emission[np.newaxis, :, obs].T, 1)
                prev[:, i] = np.argmax(F[:, i - 1] * Transition.T, 1)

        # Build the output, optimal model trajectory (path)
        path = T * [1]
        path[-1] = np.argmax(F[:, T - 1])
        for i in reversed(range(1, T)):
            path[i - 1] = int(prev[path[i], i])

        # calculate the probability of obtaining the path sequence
        P = np.amin(np.amax(F, axis=0))

        return path, P

    except BaseException:
        return None, None
