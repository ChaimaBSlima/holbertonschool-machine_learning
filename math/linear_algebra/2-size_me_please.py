#!/usr/bin/env python3
def matrix_shape(matrix):
    shape = []
    vector = matrix
    while type(vector) is list:
        shape.append(len(vector))
        vector = vector[0]
    return shape
