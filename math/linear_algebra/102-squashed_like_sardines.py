#!/usr/bin/env python3
"""
Task 17 (Advanced): Squashed Like Sardines
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


def cat_matrices(mat1, mat2, axis=0):
    """
    concatenates two matrices along a specific axis.

    Parameters:
    mat1 :
        A nested list representing the first matrix.
    mat2 :
        A nested list representing the second matrix.

    Returns:
        None: if the 2 matrix have not the same shape
        ConcatinatedMat: The result of the matrix
                        concatination of mat1 and mat2.
    """
    if len(matrix_shape(mat1)) != len(matrix_shape(mat2)):
        return (None)
    ConcatinatedMat = cat_matrices_recursion(mat1, mat2, axis)

    return (ConcatinatedMat)


def cat_matrices_recursion(mat1, mat2, axis):
    """
    concatenates two matrices along a specific axis using recursion.

    Parameters:
    mat1 :
        A nested list representing the first matrix.
    mat2 :
        A nested list representing the second matrix.

    Returns:
        None: if the 2 matrix have not the same shape
        ConcatinatedMat: The result of the matrix
                        concatination of mat1 and mat2.
    """
    ConcatinatedMat = []
    if axis == 0:
        return mat1 + mat2
    for i in range(len(mat1)):
        if type(mat1[i]) is list and type(mat2[i] is list):
            row = cat_matrices_recursion(mat1[i], mat2[i], axis - 1)
            if row is None:
                return (None)
            ConcatinatedMat.append(row)
        elif axis == 1:
            ConcatinatedMat.append(mat1[i]+mat2[i])
        else:
            return (None)
    return (ConcatinatedMat)
