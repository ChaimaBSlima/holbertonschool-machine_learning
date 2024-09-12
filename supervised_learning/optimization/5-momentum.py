#!/usr/bin/env python3
""" Task 5: 5. Momentum """
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with
    momentum optimization algorithm.

    Args:
        alpha (float):
            The learning rate.
        beta1 (float):
            The momentum weight, a value between 0 and 1 that dictates
            how much of the previous gradient to retain in the velocity
            calculation.
        var (numpy.ndarray):
            The variable to be updated (e.g., weights of the model).
        grad (numpy.ndarray): The gradient of `var` with respect to the loss.
        v (numpy.ndarray):
            The previous velocity (i.e., the exponentially weighted
            average of past gradients).

    Returns:
        tuple: A tuple containing:
            - var (numpy.ndarray): The updated variable.
            - v (numpy.ndarray): The new velocity (updated first moment).
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
