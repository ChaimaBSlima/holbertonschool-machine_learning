#!/usr/bin/env python3
""" Task 9: 9. Adam """


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.

    Args:
        alpha (float):
            The learning rate.
        beta1 (float):
            The weight used for the first moment
            (exponentially weighted average of the gradient).
        beta2 (float):
            The weight used for the second moment
            (exponentially weighted average of the squared gradient).
        epsilon (float):
            A small number to avoid division by zero.
        var (numpy.ndarray):
            The variable to be updated.
        grad (numpy.ndarray):
            The gradient of the variable.
        v (numpy.ndarray):
            The previous first moment
            (moving average of the gradient).
        s (numpy.ndarray):
            The previous second moment
            (moving average of the squared gradient).
        t (int):
            The time step for bias correction.

    Returns:
        numpy.ndarray: The updated variable.
        numpy.ndarray: The updated first moment.
        numpy.ndarray: The updated second moment.
    """
    α = alpha
    β1 = beta1
    β2 = beta2
    ε = epsilon

    Vd = (β1 * v) + ((1 - β1) * grad)
    Sd = (β2 * s) + ((1 - β2) * grad * grad)

    Vd_ok = Vd / (1 - β1 ** t)
    Sd_ok = Sd / (1 - β2 ** t)

    w = var - α * (Vd_ok / ((Sd_ok ** (0.5)) + ε))
    return (w, Vd, Sd)
