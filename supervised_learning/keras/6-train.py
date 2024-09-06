#!/usr/bin/env python3
""" Task 6: 6. Early Stopping """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent
    with optional early stopping.

    Parameters:
    -----------
    network : keras.Model
        The model to train.
    data : numpy.ndarray
        Input data of shape (m, nx), where `m` is the number of samples
        and `nx` is the number of features.
    labels : numpy.ndarray
        One-hot encoded labels of shape (m, classes), where `classes` is
        the number of categories.
    batch_size : int
        Size of the mini-batches used for gradient descent.
    epochs : int
        Number of passes through the entire dataset.
    validation_data : tuple, optional
        Data on which to evaluate the model during training.
        Should be a tuple (val_data, val_labels). Default is None.
    early_stopping : bool, optional
        Whether to apply early stopping based on validation loss.
        Default is False.
    patience : int, optional
        Number of epochs with no improvement after which training
        will stop if early stopping is enabled.
        Default is 0 (no patience).
    verbose : bool, optional
        Whether to print progress during training (default is True).
    shuffle : bool, optional
        Whether to shuffle the data before each epoch
        (default is False for reproducibility).

    Returns:
    history : History object
        A Keras History object that holds the record of training loss,
        metrics, and validation performance (if validation data is provided).
    """
    callbacks = []
    if (validation_data):
        early_stopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)
        callbacks.append(early_stopping)

    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle,
                          callbacks=callbacks)

    return history
