#!/usr/bin/env python3
"""
Task 2 and 3: Class for the Multinirmal probabilities
"""

import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal (Gaussian) distribution.

    Attributes:
    mean (numpy.ndarray): A (d, 1) array representing the mean vector
    of the distribution.
    cov (numpy.ndarray): A (d, d) array representing the covariance
    matrix.

    Methods:
    pdf(x): Computes the Probability Density Function (PDF) at a
    given data point x.
    """
    def __init__(self, data):
        """
        Initializes the MultiNormal distribution with the provided dataset.

        Parameters:
        data (numpy.ndarray): A 2D array of shape (d, n), where:
                              - d is the number of dimensions (features).
                              - n is the number of data points.

        Raises:
        TypeError: If data is not a 2D numpy array.
        ValueError: If data does not contain multiple data points.

        Notes:
        - The mean vector is computed as the average of each feature.
        - The covariance matrix is computed using the unbiased estimator.
        - The dataset is transposed so that features correspond to rows.
        """

        if (isinstance(data, type(None))):
            raise TypeError('data must be a 2D numpy.ndarray')

        if (not isinstance(data, np.ndarray)) or (len(data.shape)) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        if (data.shape[1] < 2):
            raise ValueError("data must contain multiple data points")

        data = data.T
        mean = data.mean(axis=0)
        mean = np.reshape(mean, (-1, data.shape[1]))

        n = data.shape[0] - 1
        x = data - mean
        cov = np.dot(x.T, x) / n

        self.mean = mean.T
        self.cov = cov

    def pdf(self, x):
        """
        Computes the Probability Density Function (PDF) of the
        Multivariate Normal distribution.

        Parameters:
        x (numpy.ndarray): A (d, 1) column vector representing a
        single data point.

        Returns:
        float: The probability density at point x.

        Raises:
        TypeError: If x is not a numpy array.
        ValueError: If x does not have the required shape (d, 1).

        Notes:
        - The PDF is computed using the formula:
p(x | μ, Σ) = (1 / sqrt((2π)^d * |Σ|)) * exp(-0.5 * (x - μ)^T *Σ⁻¹ * (x - μ))
        - This function assumes that the covariance matrix is invertible.

        Reference:
        - https://peterroelants.github.io/posts/multivariate-normal-primer/
            p(x∣μ,Σ)=[1√(2π)^d|Σ|] * exp((−1/2)(x−μ)^T(Σ^−1(x−μ))
        """

        d = self.cov.shape[0]

        if (isinstance(x, type(None))):
            raise TypeError('x must be a numpy.ndarray')

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if (x.shape[0] != d):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        if (len(x.shape) != 2) or (x.shape[1] != 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        mean = self.mean
        cov = self.cov

        x_m = x - mean
        pdf = (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(cov)))
               * np.exp(-(np.linalg.solve(cov, x_m).T.dot(x_m)) / 2))

        return pdf[0][0]
