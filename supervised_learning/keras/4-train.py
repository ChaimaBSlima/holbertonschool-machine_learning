#!/usr/bin/env python3
""" Task 4: 4. Train """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Parameters:
    network : keras.Model
        The model to train.
    data : numpy.ndarray
        Input data of shape (m, nx),
        where `m` is the number of samples
        and `nx` is the number of features.
    labels : numpy.ndarray
        One-hot encoded labels of shape (m, classes),
        where `classes` is the number of categories.
    batch_size : int
        Size of the mini-batches used for gradient descent.
    epochs : int
        Number of passes through the entire dataset.
    verbose : bool, optional
        Whether to print progress during training
        (default is True).
    shuffle : bool, optional
        Whether to shuffle the data before each epoch
        (default is False for reproducibility).

    """
    return network.fit(x=data,
                       y=labels,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       verbose=verbose)
