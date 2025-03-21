#!/usr/bin/env python3
""" Poisson Distribution """


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
