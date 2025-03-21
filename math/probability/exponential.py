#!/usr/bin/env python3
""" Poisson Distribution """


class Exponential:
    """
    Represents an exponential distribution.

    The exponential distribution describes the time between events in a
    Poisson process, where events occur continuously and independently
    at a constant rate.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes an Exponential distribution.

        Parameters:
        data (list, optional):
        A list of data points used to estimate the
        distribution parameter (λ). Defaults to None.
        lambtha (float, optional):
        The expected number of occurrences in
         a given time frame (λ). Defaults to 1.

        Raises:
        TypeError: If `data` is not a list.
        ValueError: If `data` has less than two values.
        ValueError: If `lambtha` is not a positive value.
        """
        self.e = 2.7182818285
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """
        Calculates the probability density function
        (PDF) at a given time `x`.

        Parameters:
        x (float):
        The time period for which the probability is calculated.

        Returns:
        float: The probability density at `x`.

        Notes:
        - If `x` is negative, the function returns 0.
        - The PDF formula is: P(x) = λ * e^(-λx)
        """
        if x < 0:
            return 0
        return self.lambtha * self.e**(-self.lambtha*x)

    def cdf(self, x):
        """
        Calculates the cumulative distribution
        function (CDF) at a given time `x`.

        Parameters:
        x (float):
        The time period for which the cumulative
        probability is calculated.

        Returns:
        float:
        The probability of an event occurring within time `x`.

        Notes:
        - If `x` is negative, the function returns 0.
        - The CDF formula is: P(X ≤ x) = 1 - e^(-λx)
        """
        if x < 0:
            return 0
        return 1 - self.e**(-self.lambtha*x)
