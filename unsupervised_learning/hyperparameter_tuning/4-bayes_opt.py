#!/usr/bin/env python3
""" Task 4: 4. Bayesian Optimization - Acquisition """
from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """ Performs Bayesian optimization on a noiseless 1D Gaussian Process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initializes the Bayesian Optimization instance.

        Parameters:
        f (function):
        The black-box function to be optimized.

        X_init (np.ndarray):
        Array of shape (t, 1) with the inputs already sampled.

        Y_init (np.ndarray):
        Array of shape (t, 1) with the outputs of the black-box function.

        bounds (tuple):
        Tuple of (min, max) representing the bounds of the search space.

        ac_samples (int):
        Number of samples to analyze during acquisition.

        l (float):
        Length parameter for the Gaussian Process kernel. Default is 1.

        sigma_f (float):
        Standard deviation of the output for the GP kernel. Default is 1.

        xsi (float):
        Exploration-exploitation factor for acquisition. Default is 0.01.

        minimize (bool):
        Whether optimization is for minimization (True) or
        maximization (False).
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.zeros((ac_samples, 1))
        self.X_s = np.linspace(start=bounds[0],
                               stop=bounds[1],
                               num=ac_samples,
                               endpoint=True)
        self.X_s = self.X_s.reshape(ac_samples, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Computes the next best sample location using the Expected
        Improvement (EI) acquisition function.

        Returns:
        tuple:
            - X_nest (np.ndarray):
              The optimal next point to sample, shape (1,).
            - EI (np.ndarray):
              Expected Improvement for each candidate sample point in self.X_s,
              shape (ac_samples,).
        """
        m_sample, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            sam = np.min(self.gp.Y)
            imp = sam - m_sample - self.xsi
        else:
            sam = np.max(self.gp.Y)
            imp = m_sample - sam - self.xsi

        with np.errstate(divide='warn'):
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_nest = self.X_s[np.argmax(EI)]
        return X_nest, EI
