#!/usr/bin/env python3
"""
Task 12:  Bracing The Elements
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction,
    multiplication, and division on two NumPy matrices.

    Parameters:
    mat1 : numpy.ndarray
        The first NumPy array for element-wise operations.
    mat2 : numpy.ndarray
        The second NumPy array for element-wise operations.

    Returns:
    tuple
        A tuple containing 4 NumPy arrays:
        - The element-wise addition of mat1 and mat2.
        - The element-wise subtraction of mat1 and mat2.
        - The element-wise multiplication of mat1 and mat2.
        - The element-wise division of mat1 and mat2.
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return (add, sub, mul, div)
