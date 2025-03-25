#!/usr/bin/env python3
""" Task 3: 3. Posterior """

import numpy as np


def n_choose_x(n, x):
    """
    Computes the binomial coefficient (n choose x).

    Args:
        n (int): The total number of trials.
        x (int): The number of successful trials.

    Returns:
        float: The computed binomial coefficient.
    Raises:
        ValueError: If n or x is negative or non-integer.
    """
    n_fact = np.math.factorial(n)
    x_fact = np.math.factorial(x)
    nx_fact = np.math.factorial(n - x)
    return n_fact / (x_fact * nx_fact)


def likelihood(x, n, P):
    """
    Computes the likelihood of obtaining exactly x successes in n trials
    given a probability P of success on a single trial.

    Args:
        x (int): Number of successful trials.
        n (int): Total number of trials.
        P (numpy.ndarray): 1D array of probabilities of success for each trial.

    Returns:
        numpy.ndarray: The likelihood of observing x successes for
        each probability in P.

    Raises:
        ValueError: If n is not a positive integer.
        ValueError: If x is not an integer >= 0.
        ValueError: If x > n.
        TypeError: If P is not a 1D numpy.ndarray.
        ValueError: If any value in P is outside the range [0, 1].
    """

    if (not isinstance(n, (int, float)) or n <= 0):
        raise ValueError("n must be a positive integer")

    if (not isinstance(x, (int, float)) or x < 0):
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)

    if (x > n):
        raise ValueError("x cannot be greater than n")

    if (not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1):
        raise TypeError("P must be a 1D numpy.ndarray")

    if (np.any(P > 1) or np.any(P < 0)):
        raise ValueError("All values in P must be in the range [0, 1]")

    binomial_coef = n_choose_x(n, x)
    success_rate = pow(P, x)
    failure_rate = pow(1 - P, n - x)

    likelihood = binomial_coef * success_rate * failure_rate

    return likelihood


def intersection(x, n, P, Pr):
    """
        Computes the intersection probability of events.

    Args:
        x (int): Number of successful trials.
        n (int): Total number of trials.
        P (numpy.ndarray): 1D array of probabilities of success for each trial.
        Pr (numpy.ndarray): 1D array of prior probabilities of P.

    Returns:
        numpy.ndarray: The intersection probability of events.
    Raises:
        TypeError: If P or Pr is not a 1D numpy.ndarray.
        TypeError: If Pr does not have the same shape as P.
        ValueError: If any value in Pr is outside the range [0, 1].
        ValueError: If Pr does not sum to 1.
    """

    if (not isinstance(Pr, np.ndarray)):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if (not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1):
        raise TypeError("P must be a 1D numpy.ndarray")

    if (Pr.shape != P.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if (np.any(Pr > 1) or np.any(Pr < 0)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if (not np.isclose(np.sum(Pr), 1)):
        raise ValueError("Pr must sum to 1")

    # If events are independent , P(A ∩ B) = P(A) * P(B)
    # If not, P(A ∩ B) = P(B∣A) * P(A) / P(B)

    intersection = Pr * likelihood(x, n, P)
    return intersection


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of x successes in n trials.

    Parameters:
    x (int): Number of successes.
    n (int): Total number of trials.
    P (numpy.ndarray): 1D array of hypothetical probabilities of success.
    Pr (numpy.ndarray): 1D array of prior probabilities of P.

    Returns:
    float: The marginal probability of x successes in n trials.
    """
    if not isinstance(n, int) or (n <= 0):
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or (x < 0):
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)

    if (x > n):
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    if not isinstance(Pr, np.ndarray):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if (Pr.shape != P.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not (np.isclose(np.sum(Pr), 1)):
        raise ValueError("Pr must sum to 1")

    return np.sum(intersection(x, n, P, Pr))


def posterior(x, n, P, Pr):
    """
    Computes the posterior probability of P given x and n using Bayes' theorem.

    Parameters:
        x (int): Number of successes.
        n (int): Total trials.
        P (numpy.ndarray): Array of hypothetical probabilities.
        Pr (numpy.ndarray): Prior probabilities associated with each P.

    Returns:
        numpy.ndarray: Posterior probabilities.
    """
    i = intersection(x, n, P, Pr)

    return i / np.sum(i)
