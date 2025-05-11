#!/usr/bin/env python3
""" Task 4: 4. Deep RNN """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Forward propagation through a deep RNN for a sequence over multiple layers.

    Parameters:
        rnn_cells (list of RNNCell): List of RNNCell (or subclasses) instances,
            one per layer.
        X (np.ndarray): Input data, shape (t, m, i), where
            t is number of time steps,
            m is batch size,
            i is input dimensionality.
        h_0 (np.ndarray): Initial hidden states, shape (l, m, h), where
            l is number of layers and
            h is hidden state dimensionality.

    Returns:
        H (np.ndarray): Hidden states for all layers and time steps,
            shape (t + 1, l, m, h). H[0] is the initial states h_0.
        Y (np.ndarray): Outputs for each time step,
            shape (t, m, o), where o is output dimensionality.
    """

    h_prev = h_0
    H = np.array(([h_prev]))
    H = np.repeat(H, X.shape[0] + 1, axis=0)

    for i in range(X.shape[0]):
        for a_layer, cell in enumerate(rnn_cells):

            # forwarding
            parameter = X[i] if a_layer == 0 else h_prev
            h_prev, y = cell.forward(H[i, a_layer], parameter)

            # update the hidden states
            H[i + 1, a_layer] = h_prev

            # update all the outputs
            if (i != 0):
                Y[i] = y

            else:
                Y = np.array([y])
                Y = np.repeat(Y, X.shape[0], axis=0)

    return H, Y
