#!/usr/bin/env python3
""" Task 1: 1. Gaussian Process Prediction """
import numpy as np


class GaussianProcess():
    """ Represents a noiseless 1D Gaussian process. """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor.

        Parameters:
        - X_init (numpy.ndarray): shape (t, 1),
          representing the inputs already sampled with the
          black-box function.
        - Y_init (numpy.ndarray): shape (t, 1),
          representing the outputs of the black-box function
          for each input in X_init.
        - l (float): the length parameter for the kernel. Default is 1.
        - sigma_f (float): the standard deviation given to the output of the
          black-box function. Default is 1.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        using the Radial Basis Function (RBF).

        Parameters:
        - X1 (numpy.ndarray): shape (m, 1)
        - X2 (numpy.ndarray): shape (n, 1)

        Returns:
        - numpy.ndarray: shape (m, n), the covariance kernel matrix.
        """
        # K(xᵢ, xⱼ) = σ² exp((-0.5 / 2l²)(xᵢ − xⱼ)ᵀ (xᵢ − xⱼ))
        σ2 = self.sigma_f ** 2
        l2 = self.l ** 2

        sqr_sumx1 = np.sum(X1**2, 1).reshape(-1, 1)
        # print("sqr_sum1", sqr_sumx1)
        sqr_sumx2 = np.sum(X2**2, 1)
        # print("sqr_sum2", sqr_sumx2)
        sqr_dist = sqr_sumx1 - 2 * np.dot(X1, X2.T) + sqr_sumx2

        kernel = σ2 * np.exp(-0.5 / l2 * sqr_dist)
        return kernel

    def predict(self, X_s):
        """
        Predicts the mean and variance of points in a Gaussian Process.

        Parameters:
        X_s (np.ndarray): Array of shape (s, 1) containing the sample locations
                          where predictions will be made.

        Returns:
        tuple:
            - μ (np.ndarray): Mean vector of the predicted distribution
                              for each point in X_s, shape (s,).
            - cov_s (np.ndarray): Variances (diagonal of the covariance matrix)
                                  for each point in X_s, shape (s,).
        """
        s = X_s.shape[0]
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s) + np.ones(s) - np.eye(s)
        K_inv = np.linalg.inv(K)

        μ = (K_s.T.dot(K_inv).dot(self.Y)).flatten()

        cov_s = (K_ss - K_s.T.dot(K_inv).dot(K_s))
        cov_s = np.diag(cov_s)

        return (μ, cov_s)
