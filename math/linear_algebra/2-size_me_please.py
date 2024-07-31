#!/usr/bin/env python3
"""
Task 2: 2. Size Me Please
"""


def matrix_shape(matrix):
    """
    Calculate the shape of a matrix.

    Args:
        matrix (list): A nested list representing the matrix.

    Returns:
        list: A list of integers representing the shape of the matrix.
              The length of this list represents the number of dimensions.
    """
    shape = []
    vector = matrix
    while type(vector) is list:
        shape.append(len(vector))
        vector = vector[0]
    return shape
