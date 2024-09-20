#!/usr/bin/env python3
""" Task 0: 0. L2 Regularization Cost """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (float):The original cost of the network without regularization.
        lambtha (float): The regularization parameter (lambda).
        weights (dict): A dictionary of the weights of the neural network.
                        Each key represents a layer
                        (e.g., "W1", "W2", ..., "WL").
        L (int): The number of layers in the neural network.
        m (int): The number of data points.

    Returns:
        float:
        The cost of the neural network including the L2 regularization term.
    """
    summation = 0
    for ly in range(1, L + 1):
        key = "W{}".format(ly)
        summation += np.linalg.norm(weights[key])

    L2_cost = lambtha * summation / (2 * m)

    return cost + L2_cost
