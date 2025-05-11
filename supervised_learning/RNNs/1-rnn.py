#!/usr/bin/env python3
""" Task 1: 1. RNN """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
        Perform forward propagation through an entire RNN sequence.

    Parameters:
        rnn_cell (RNNCell): Instance of RNNCell to use for each time step.
        X (np.ndarray): Input data for the cell, shape (t, m, i), where
                        t is number of time steps,
                        m is batch size,
                        i is dimensionality of input data.
        h_0 (np.ndarray): Initial hidden state, shape (m, h).

    Returns:
        H (np.ndarray): Hidden states for each time step, shape (t + 1, m, h).
                        H[0] is h_0.
        Y (np.ndarray): Outputs for each time step, shape (t, m, o).
    """
    h_prev = h_0
    H = np.array(([h_0]))

    t = X.shape[0]
    for i in range(t):
        h_prev, y = rnn_cell.forward(h_prev, X[i])

        H = np.append(H, [h_prev], axis=0)
        Ŷ = np.array([y]) if i == 0 else np.append(Ŷ, [y], axis=0)

    return H, Ŷ
