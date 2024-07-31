#!/usr/bin/env python3
"""
Task 5: 5. Across The Planes
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
    if matrix_shape(mat1) != matrix_shape(mat2):
        return (None)
    SommedMat = []
    for i in range(len(mat1)):
        SommedMat.append([])
        for j in range(len(mat1[0])):
            SommedMat[i].append(mat1[i][j] + mat2[i][j])
    return SommedMat
