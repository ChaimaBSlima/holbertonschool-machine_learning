#!/usr/bin/env python3
"""task 10: 10. Derive happiness in oneself from a good day's work"""
import numpy as np


def poly_derivative(poly):
    """
     Calculate the derivative of a polynomial.

    Args:
        n (int): The upper limit of the summation.oly (list): A list of numbers
        (int or float) representing the polynomial coefficients.

    Returns:
        list: A list of numbers representing the derivative of the polynomial.
        None: If the input is not a list.
    """
    if not isinstance(poly, list):
        return None
    result = []
    for i in poly[1:]:
        result.append(poly[i]*i)
    return result
