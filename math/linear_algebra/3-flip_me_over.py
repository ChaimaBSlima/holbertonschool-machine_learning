#!/usr/bin/env python3
"""
Task 3: 3. Flip Me Over
"""


def matrix_transpose(matrix):
    """
    Find the transpose of a 2D matrix.

    Args:
        matrix (list): A nested list representing the matrix.

    Returns:
        matrix(list):  the transpose of a 2D matrix.
    """
    transpose = []
    j = 0
    while j < len(matrix[0]):
        transpose.append([])
        i = 0
        while i < len(matrix):
            """print(matrix[i][j],i,j,len(matrix), len(matrix[0]))
            : to debug the code"""
            transpose[j].append(matrix[i][j])
            i += 1
        j += 1
    return (transpose)
