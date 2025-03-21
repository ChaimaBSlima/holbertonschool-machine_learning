#!/usr/bin/env python3
""" Task 5: 5. Definiteness """

import numpy as np


def definiteness(matrix):
    """
    Determines the definiteness of a square matrix.

    Parameters:
    matrix (numpy.ndarray): A square matrix of shape
    (n, n) whose definiteness should be determined.

    Returns:
    str or None: A string describing the definiteness of the matrix:
        - 'Positive definite' if all eigenvalues are positive.
        - 'Positive semi-definite' if all eigenvalues are
            non-negative.
        - 'Negative definite' if all eigenvalues are negative.
        - 'Negative semi-definite' if all eigenvalues are
            non-positive.
        - 'Indefinite' if the matrix has both positive and
            negative eigenvalues.
        - None if the matrix is not square or not symmetric.
    
    Raises:
    TypeError:
    If the input is not a numpy.ndarray.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')

    l = matrix.shape[0]
    if len(l) != 2 or l != matrix.shape[1]:
        return None

    transpose = np.transpose(matrix)
    if not np.array_equal(transpose, matrix):
        return None

    w, v = np.linalg.eig(matrix)

    if all(w > 0):
        return 'Positive definite'
    elif all(w >= 0):
        return 'Positive semi-definite'
    elif all(w < 0):
        return 'Negative definite'
    elif all(w <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
