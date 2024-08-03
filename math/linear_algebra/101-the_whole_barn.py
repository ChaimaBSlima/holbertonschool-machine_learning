#!/usr/bin/env python3
"""
Task 16 (Advanced): The Whole Barn
"""


def add_matrices(mat1, mat2):
    """
    Adds two matrices.

    Parameters:
    mat1 :
        A nested list representing the first matrix
    mat2 :
        A nested list representing the second matrix

    Returns:
        None: if the 2 matrix have not the same shape
        The result of the matrix addition of mat1 and mat2.
    """
    if len(mat1) != len(mat2):
        return (None)
    AddedMat = []
    for i in range(len(mat1)):
        if type(mat1[i]) is list and type(mat2[i] is list):
            item = add_matrices(mat1[i], mat2[i])
            if item is None:
                return (None)
            AddedMat.append(item)
        else:
            AddedMat.append(mat1[i]+mat2[i])
    return (AddedMat)
