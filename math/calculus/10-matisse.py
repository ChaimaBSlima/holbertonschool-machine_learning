#!/usr/bin/env python3
"""task 10: 10. Derive happiness in oneself from a good day's work"""


def poly_derivative(poly):
    """
     Calculate the derivative of a polynomial.

    Args:
        poly (list): A list of numbers (int or float)
        representing the polynomial coefficients.

    Returns:
        list: A list of numbers representing the derivative of the polynomial.
        None: If the input is not a list.
    """
    if not isinstance(poly, list) or poly == []:
        return None
    if len(poly) == 1:
        result = [0]
    else:
        result = []
        for i in range(1, len(poly)):
            result.append(poly[i]*i)
    return result
