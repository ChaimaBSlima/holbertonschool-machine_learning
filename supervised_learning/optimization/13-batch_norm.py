#!/usr/bin/env python3
""" Task 13: 13. Batch Normalization """


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using
    batch normalization.

    Args:
        Z (numpy.ndarray):
            The input data of shape (m, n) to be normalized,
            where m is the number of data points and n is the number of features.
        gamma (numpy.ndarray):
            The scales used for batch normalization, shape (1, n).
        beta (numpy.ndarray):
            The offsets used for batch normalization, shape (1, n).
        epsilon (float):
            A small number to avoid division by zero.

    Returns:
        numpy.ndarray: The batch-normalized output, same shape as Z.
    """
    β = beta
    γ = gamma
    ε = epsilon

    μ = Z.mean(0)
    σ = Z.std(0)
    σ2 = Z.std(0) ** 2

    z_normalized = (Z - μ) / ((σ2 + ε) ** (0.5))
    Ẑ = γ * z_normalized + β

    return Ẑ
