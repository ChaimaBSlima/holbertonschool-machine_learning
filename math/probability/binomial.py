#!/usr/bin/env python3
""" Binomial Distribution """


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


class Binomial:
    """
    Class representing a binomial distribution.

    The binomial distribution models the number of successes in a
    fixed number of independent Bernoulli trials.
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution parameters.

        If data is provided, estimates the distribution parameters (n, p)
        based on the sample data.

        Parameters:
        data (list, optional):
            A dataset used to estimate the distribution parameters.
            Defaults to None.
        n (int, optional):
            The number of Bernoulli trials. Defaults to 1.
        p (float, optional):
            The probability of success in each trial. Defaults to 0.5.

        Raises:
        TypeError: If data is not a list.
        ValueError: If data contains fewer than two values.
        ValueError: If n is not a positive integer.
        ValueError: If p is not in the range (0, 1).
        """
        if data is not None:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = float(sum(data)/len(data))
            new_data = [(x - mean) ** 2 for x in data]
            variance = sum(new_data) / len(data)
            p = 1 - variance / mean
            if ((mean / p) - (mean // p)) >= 0.5:
                self.n = 1 + int(mean / p)
            else:
                self.n = int(mean / p)
            self.p = float(mean / self.n)
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if (p <= 0) or (p >= 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
            self.n = int(n)

    def pmf(self, k):
        """
        Computes the Probability Mass Function (PMF) for
        a given number of successes.

        The PMF gives the probability of observing exactly
        k successes in n trials.

        Parameters:
        k (int):
            The number of successes.

        Returns:
        float: The probability of exactly k successes occurring.
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0

        comb = factorial(self.n) / (factorial(self.n-k) * factorial(k))
        return comb * self.p**k * ((1-self.p)**(self.n-k))

    def cdf(self, k):
        """
        Computes the Cumulative Distribution Function (CDF
        up to k successes.

        The CDF gives the probability of observing up to
        and including k successes.

        Parameters:
        k (int):
            The number of successes.

        Returns:
        float: The probability of k or fewer successes occurring.
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0

        cdf = 0
        for i in range(k+1):
            cdf += self.pmf(i)
        return cdf
