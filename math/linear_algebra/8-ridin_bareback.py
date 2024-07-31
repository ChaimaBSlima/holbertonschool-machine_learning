#!/usr/bin/env python3
"""
Task 8: Ridin’ Bareback
"""


def mat_mul(mat1, mat2):
    """
    perform matrix multiplication

    Args:
        mat1: the first matrice.
        mat2: the second matrice.

    Returns:
        None: if the two matrices cannot be multiplied.
        MultiplicatedMat: a new matrix that is the result of multiplication.

    """
    if len(mat1[0]) != len(mat2):
        return None
    MultiplicatedMat = [[0 for _ in range(len(mat2[0]))]
                        for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                MultiplicatedMat[i][j] = MultiplicatedMat[i][j]\
                    + mat1[i][k] * mat2[k][j]
    return (MultiplicatedMat)
