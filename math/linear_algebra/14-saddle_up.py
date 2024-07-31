#!/usr/bin/env python3
"""
Task 14:Saddle Up
"""


def np_matmul(mat1, mat2):
    """
    Performs  matrix multiplication.

    Parameters:
    mat1 : numpy.ndarray
        The first NumPy array for element-wise operations.
    mat2 : numpy.ndarray
        The second NumPy array for element-wise operations.

    Returns:
        The result of the matrix multiplication of mat1 and mat2.
    """
    return (mat1 @ mat2)
