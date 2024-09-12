#!/usr/bin/env python3
""" Task 7: 7. RMSProp """
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon,
                             var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float):
            The learning rate, controlling the step size during updates.
        beta2 (float):
            The RMSProp decay factor (usually between 0.9 and 0.999),
            which controls the contribution of the moving average of
            squared gradients.
        epsilon (float):
            A small constant (e.g., 1e-8) added to the denominator
            for numerical stability to avoid division by zero.
        var (numpy.ndarray):
            The variable to be updated (e.g., weights or biases).
        grad (numpy.ndarray):
            The gradient of the variable `var` with respect to the loss.
        s (numpy.ndarray):
            moving average of the squared gradient, representing the
            second moment (squared gradients) of `var`.

    Returns:
        tuple: A tuple containing:
            - updated_var (numpy.ndarray):
                The updated value of `var`.
            - new_s (numpy.ndarray):
                The updated moving average of the squared gradients.
    """
    α = alpha
    β2 = beta2
    ε = epsilon

    α = alpha
    dw = grad
    w = var

    s_new = β2 * s + (1 - β2) * (dw * dw)
    W = w - α * (dw / ((s_new ** 0.5) + ε))

    return W, s_new
