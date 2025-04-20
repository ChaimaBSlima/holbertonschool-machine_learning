#!/usr/bin/env python3
""" Task 5: 5. Bayesian Optimization """
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

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function using Bayesian Optimization.

        Parameters:
        iterations (int):
        The maximum number of iterations to perform. Default is 100.

        Returns:
        tuple:
            - X_opt (np.ndarray):
            The optimal input found during optimization, shape (1,).
            - Y_opt (np.ndarray):
            The function value corresponding to X_opt.
        """
        X_opt = 0
        Y_opt = 0

        for _ in range(iterations):
            # Find the next best sample
            X_next = self.acquisition()[0]

            # if X_next already sampled in gp.X, ommit
            if (X_next in self.gp.X):
                break

            else:
                # get Y_next, evaluate X_next is black box function
                Y_next = self.f(X_next)

                # updates a GP
                self.gp.update(X_next, Y_next)

                # if minimizing save the least otherwise save the largest
                if (Y_next < Y_opt) and (self.minimize):
                    X_opt, Y_opt = X_next, Y_next

                if not self.minimize and Y_next > Y_opt:
                    X_opt, Y_opt = X_next, Y_next

        # removing last element
        self.gp.X = self.gp.X[:-1]

        return X_opt, Y_opt
