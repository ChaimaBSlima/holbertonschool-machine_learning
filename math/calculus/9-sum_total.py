#!/usr/bin/env python3
"""task 9: 9. Our life is the sum total of all the decisions
we make every day, and those decisions are determined by our priorities"""
import numpy as np


def summation_i_squared(n):
    """
    Calculate the summation of squares from 1 to n.

    Args:
        n (int): The upper limit of the summation.

    Returns:
        int: The summation of squares from 1 to n.
        None: If the input is not an integer.
    """
    if not isinstance(n, int) or n <= 0:
        return None
    elements = np.square(np.arange(1, n+1))
    Sum = sum(elements)
    return Sum
