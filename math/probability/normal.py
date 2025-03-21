#!/usr/bin/env python3
""" Normal Distribution """


class Normal:
    """
    Represents a normal (Gaussian) distribution.

    The normal distribution is a probability distribution
    that is symmetric around the mean, with data near the
    mean more frequent than data further away.
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes a Normal distribution.

        Parameters:
        data (list, optional):
        A list of data points used to estimate the
        distribution parameters. Defaults to None.
        mean (float, optional):
        The mean (μ) of the distribution. Defaults to 0.
        stddev (float, optional):
        The standard deviation (σ) of the distribution.
        Defaults to 1.

        Raises:
        TypeError: If `data` is not a list.
        ValueError: If `data` has fewer than two values.
        ValueError: If `stddev` is not a positive value.
        """
        self.e = 2.7182818285
        self.pi = 3.1415926536

        if data is not None:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            new_list = []
            for i in data:
                new_list.append((i-self.mean)**2)
            self.stddev = (sum(new_list) / len(data)) ** 0.5
        else:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = mean
            self.stddev = stddev

    def z_score(self, x):
        """
        Calculates the z-score of a given value `x`.

        Parameters:
        x (float):
        The data point for which the z-score is calculated.

        Returns:
        float:
        The corresponding z-score.

        Formula:
        Z = (X - μ) / σ
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the data value corresponding to a given z-score.

        Parameters:
        z (float): The z-score.

        Returns:
        float: The corresponding x-value.

        Formula:
        X = Z * σ + μ
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculates the probability density
        function (PDF) at a given `x`.

        Parameters:
        x (float): The data point.

        Returns:
        float: The probability density at `x`.

        Formula:
        f(x) = (1 / (σ√(2π))) * e^(-((x-μ)² / (2σ²)))
        """
        return (self.e ** -((x-self.mean)**2 / (2*self.stddev**2)))\
            / (self.stddev * (2*self.pi)**.5)

    def cdf(self, x):
        """
        Calculates the cumulative distribution function
        (CDF) at a given `x`.

        Parameters:
        x (float): The data point.

        Returns:
        float: The cumulative probability up to `x`.

        Approximation using the Taylor series expansion
        of the error function (erf):
        Φ(x) = (1/2) * (1 + erf((x - μ) / (σ√2)))

        where:
        erf(y) ≈ (2 / √π) * (y - (y³ / 3) + (y⁵ / 10)
                - (y⁷ / 42) + (y⁹ / 216))
        """
        arg = (x - self.mean) / (self.stddev * 2 ** 0.5)
        erf = (2 / 3.1415926536 ** 0.5) * \
              (arg - (arg ** 3) / 3 + (arg ** 5) / 10 -
               (arg ** 7) / 42 + (arg ** 9) / 216)
        return (1/2) * (1 + erf)
