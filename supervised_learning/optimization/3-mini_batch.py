#!/usr/bin/env python3
""" Task 3: 3. Mini-Batch """
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training
    a neural network using mini-batch gradient descent

    Parameters:
    X: numpy.ndarray(m, nx) representing input data
        m is the number of data points
        nx is the number of features
    Y: numpy.ndarray(m, ny) representing the labels
        m is the same number of data points as in X
        ny is the number of features
    batch_size: int
        number of data points in a batch
    Return:
    list of mini-batches containing tuples (X_batch, Y_batch)
    """
    x_shuffled, y_shuffled = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = list()
    if batch_size > m:
        batch_size = m
    for i in range(0, m, batch_size):
        X_batch = x_shuffled[i:i + batch_size]
        Y_batch = y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches
