#!/usr/bin/env python3
""" Poisson Distribution """


def factorial(n):
    """
    Calculates the factorial of a given number.

    Parameters:
    n (int):
    The number for which the factorial is to be computed.

    Returns:
    int: The factorial of `n`.

    Notes:
    - The factorial of 0 is defined as 1.
    - The function uses an iterative approach.
    """
    if n == 0:
        return 1

    fact = 1

    for i in range(1, n+1):
        fact = fact * i
    return fact


class Poisson:
    """ Represents the Posisson distribution """

    def __init__(self, data=None, lambtha=1):
        """
        Initializes a Poisson distribution.

        Parameters:
        data (list, optional):
        A list of data points to estimate the distribution.
        If provided, `lambtha` is calculated from the data.
        lambtha (float, optional):
        The expected number of occurrences in a given time frame.
        Defaults to 1. Must be positive if `data` is not provided.

        Raises:
        TypeError: If `data` is not a list.
        ValueError: If `data` contains fewer than two values.
        ValueError: If `lambtha` is not positive.
        """
        self.e = 2.7182818285

        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pmf(self, k):
        """
        Calculates the Probability Mass Function (PMF)
        for the Poisson distribution.

        Parameters:
        k (int or float):
        The number of occurrences (successes).
        If a float is provided, it is cast to an integer.

        Returns:
        float: The probability of observing `k` occurrences.

        Notes:
        - If `k` is negative, the function returns 0.
        - The PMF formula used:
          P(k) = (e^(-λ) * λ^k) / k!
        """
        if k < 0:
            return 0
        if type(k) is not int:
            k = int(k)
        summation = ((self.e**-self.lambtha) *
                     (self.lambtha**k)) / factorial(k)
        return summation

    def cdf(self, k):
        """
        Calculates the Cumulative Distribution Function (CDF)
        for a given number of occurrences.

        Parameters:
        k (int or float):
        The number of occurrences (successes). If `k` is a float,
        it is converted to an integer.

        Returns:
        float:
        The cumulative probability of obtaining at most
        `k` occurrences.

        Notes:
        - If `k` is negative, the function returns 0.
        - The CDF is computed as the sum of the PMF
            values from 0 to `k`:
          CDF(k) = Σ P(i) for i = 0 to k
        """
        if k < 0:
            return 0
        if type(k) is not int:
            k = int(k)
        summation = 0
        for i in range(0, k+1):
            summation = summation + self.pmf(i)
        return summation
