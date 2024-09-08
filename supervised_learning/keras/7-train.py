#!/usr/bin/env python3
""" Task 7: 7. Learning Rate Decay """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with optional
    early stopping and learning rate decay.

    Args:
        network (keras.Model):
          The model to train.
        data (numpy.ndarray):
          Input data of shape (m, nx), where `m` is the number
          of samples and `nx` is the number of features.
        labels (numpy.ndarray):
          One-hot encoded labels of shape (m, classes).
        batch_size (int):
          Size of the mini-batches used for gradient descent.
        epochs (int):
          Number of passes through the dataset.
        validation_data (tuple, optional):
          Data to evaluate the model during training.
        early_stopping (bool, optional):
          Whether to apply early stopping.
        patience (int, optional):
          Number of epochs without improvement before stopping.
        learning_rate_decay (bool, optional)
          Whether to apply learning rate decay.
        alpha (float, optional):
          Initial learning rate (default 0.1).
        decay_rate (float, optional):
          Rate of decay (default 1).
        verbose (bool, optional):
          Whether to print progress during training (default True).
        shuffle (bool, optional):
          Whether to shuffle the data before each epoch (default False).

    Returns:
        history: Keras History object with training metrics.
    """
    def learning_rate(epochs):
        """
        Updates the learning rate using inverse time decay.

        This function implements a learning rate decay schedule
        where the learning rate decreases over time according to
        the following formula:

            lr = alpha / (1 + decay_rate * epoch)

        Args:
            epoch (int): The current epoch number during training.
            As `epoch` increases, the learning rate decreases.

        Returns:
            float: The updated learning rate for the current epoch.
        """
        return alpha / (1 + decay_rate * epochs)

    callbacks = []
    if (validation_data and learning_rate_decay):
        early_stopping = K.callbacks.LearningRateScheduler(learning_rate, 1)
        callbacks.append(early_stopping)

    if (validation_data and early_stopping):
        early_stopping = K.callbacks.EarlyStopping(patience=patience)
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
