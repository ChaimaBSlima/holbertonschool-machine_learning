#!/usr/bin/env python3
"""
Task 13: Cat's Got Your Tongue
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Parameters:
    mat1 : numpy.ndarray
        The first NumPy array for element-wise operations.
    mat2 : numpy.ndarray
        The second NumPy array for element-wise operations.

    Returns:
        The concatenated matrix.
    """
    return (np.concatenate((mat1, mat2), axis))
