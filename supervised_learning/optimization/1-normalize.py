#!/usr/bin/env python3
""" Task 1: 1. Normalize """
import numpy as np


def normalize(X, m, s):
    """ 
    Normalizes (standardizes) a matrix by subtracting the mean
    and dividing by the standard deviation.

    Args:
        X (numpy.ndarray): A 2D array of shape (d, nx) to normalize, where:
            - d is the number of data points.
            - nx is the number of features.
        m (numpy.ndarray): A 1D array of shape (nx,) that contains the mean
        of all features of X.
        s (numpy.ndarray): A 1D array of shape (nx,) that contains the standard
        deviation of all features of X.

    Returns:
        numpy.ndarray: A normalized (standardized) version of X with the same
        shape as the input.
    """
    Z = (X - m) / s
    return Z
