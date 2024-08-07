#!/usr/bin/env python3
"""task 9: 9. Our life is the sum total of all the decisions
we make every day, and those decisions are determined by our priorities"""


def summation_i_squared(n):
    """
    Calculate the summation of squares from 1 to n.

    Args:
        n (int): The upper limit of the summation.

    Returns:
        int: The summation of squares from 1 to n.
        None: If the input is not an integer.
    """
    if type(n) is not int:
        return (None)
    sum = 0
    for i in range(1, n+1):
        sum = sum + (i**2)
    return sum
