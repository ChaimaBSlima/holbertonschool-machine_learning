#!/usr/bin/env python3
""" Task 4:4. Moving Average """
import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.

    Args:
        data (list or np.ndarray): A list or array containing the
        data points
            to calculate the moving average from.
        beta (float): The weight used for the moving average,
                where 0 < beta < 1.
            A higher beta puts more emphasis on recent data points.

    Returns:
        np.ndarray: The weighted moving average of the input data as
        a numpy array.
    """
    newData = []
    value = 0
    for i in range(len(data)):
        value = beta * value + (1 - beta) * data[i]
        newData.append(value/(1 - beta ** (i + 1)))

    return newData
