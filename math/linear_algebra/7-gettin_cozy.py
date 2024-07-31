#!/usr/bin/env python3
"""
Task 7: 7. Gettin’ Cozy
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    concatenate two matrices along a specific axis

    Args:
        mat1: the first matrice.
        mat2: the second matrice.

    Returns:
        None: if the two matrices cannot be concatenated.
        ConcatenatedMat: a new matrix that is the result of concatenating.

    """
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        ConcatenatedMat = []
        for i in range(len(mat1)):
            ConcatenatedMat.append(mat1[i] + mat2[i])
    else:
        if len(mat1[0]) != len(mat2[0]):
            return None
        ConcatenatedMat = [i[:] for i in mat1] + [i[:] for i in mat2]
    return ConcatenatedMat
