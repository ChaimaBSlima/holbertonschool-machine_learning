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
    X_batch, Y_batch = shuffle_data(X, Y)
    chunk = np.arange(batch_size, len(X), batch_size)
    return zip(np.array_split(X_batch, chunk, axis=0),
               np.array_split(Y_batch, chunk, axis=0))
