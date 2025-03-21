#!/usr/bin/env python3
""" Task 2: 2-cofactor.py """


def determinant(matrix):
    """
    Computes the determinant of a square matrix using recursion.

    param matrix:
    A list of lists representing the square matrix whose determinant
    is to be calculated. It must be a square matrix (n x n),
    where each inner list represents a row.

    raises TypeError:
    If the input is not a list of lists or if any row is not a list.
    raises ValueError:
    If the matrix is not square (i.e., rows have different lengths).

    return:
    The determinant of the matrix as a number (int or float).
    """

    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix[0]) == 0:
        return 1

    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0] if matrix[0] else 1

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    return sum(
        matrix[0][i] * (-1) ** i * determinant([row[:i] + row[i + 1:] for
                                                row in matrix[1:]])
        for i in range(len(matrix))
    )


def minor(matrix):
    """
    Computes the minor of a square matrix.

    param matrix: list of lists (square matrix)
    The matrix for which minors are to be calculated.
    It must be a square matrix (n x n),
    where each inner list represents a row.

    return: list of lists
    A matrix where each element is the minor of the corresponding
    element in the input matrix. The minor is calculated by
    removing the corresponding row and column and computing the
    determinant of the remaining submatrix.

    raises ValueError:
    If the input matrix is not a square matrix or is empty.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')

    for element in matrix:
        if not isinstance(element, list):
            raise TypeError('matrix must be a list of lists')

    if len(matrix) == 1 and len(matrix[0]) == 0:
        raise ValueError('matrix must be a non-empty square matrix')

    for element in matrix:
        if len(element) != len(matrix):
            raise ValueError('matrix must be a non-empty square matrix')
    if len(matrix) == 1:
        return [[1]]

    minor = []
    for i in range(len(matrix)):
        minor.append([])
        for j in range(len(matrix)):
            rows = [matrix[m] for m in range(len(matrix)) if m != i]
            new_m = [[row[n] for n in range(len(matrix)) if n != j]
                     for row in rows]
            my_det = determinant(new_m)
            minor[i].append(my_det)

    return minor


def cofactor(matrix):
    """
   Computes the cofactor matrix of a given square matrix.

    Parameters:
    matrix (list of lists):
    A square matrix represented as a list of lists.

    Returns:
    list of lists:
    The cofactor matrix of the input matrix.
    """

    M = minor(matrix)

    cofactor = []
    for i in range(len(matrix)):
        cofactor .append([])
        for j in range(len(matrix)):
            sign = (-1) ** (i + j)
            value = sign * M[i][j]
            cofactor[i].append(value)

    return cofactor
