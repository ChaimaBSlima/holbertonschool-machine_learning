#!/usr/bin/env python3
"""
Task 5: 5. Across The Planes
"""


def add_matrices2D(mat1, mat2):
    """
     Add two matrices element-wise.

    Args:
        mat1: the first matrice.
        mat2: the second matrice.

    Returns:
        None: if mat1 and mat2 are not the same shape.
        SommedArr: the new matrice with sommed values.
    """
    if not mat1 or not mat2 or not mat1[0] or not mat2[0]:
        return
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[1]):
        return(None)
    SommedMat = []
    for i in range(len(mat1)):
        SommedMat.append([])
        for j in range(len(mat1[0])):
            SommedMat[i].append(mat1[i][j] + mat2[i][j])
    return SommedMat
