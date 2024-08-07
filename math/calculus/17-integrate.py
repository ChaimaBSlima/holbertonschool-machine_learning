#!/usr/bin/env python3
"""task 17: 17. Integrate"""


def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial with an integration constant.

    Args:
         poly (list): A list of numbers (int or float)
         representing the polynomial coefficients.
         C (int, optional): The constant of integration. Defaults to 0.

    Returns:
        list: A list of numbers representing the integral of the polynomial.
        None: If the input is not a list or the constant `C` is not an integer.
    """
    if not isinstance(poly, list) or poly == [] or not isinstance(C, int):
        return None

    result = [C]
    for i in range(len(poly)):
        element = poly[i]/(i+1)
        if element.is_integer():
            result.append(int(element))
        else:
            result.append(element)
    return result
