#!/usr/bin/env python3
""" Task 2: 2. GRU Cell """
import numpy as np


class GRUCell:
    """
    Gated Recurrent Unit cell.

    Attributes:
        Wz (np.ndarray): Update gate weight, shape (i + h, h).
        Wr (np.ndarray): Reset gate weight, shape (i + h, h).
        Wh (np.ndarray):
        Candidate hidden state weight, shape (i + h, h).
        Wy (np.ndarray): Output weight, shape (h, o).
        bz (np.ndarray): Update gate bias, shape (1, h).
        br (np.ndarray): Reset gate bias, shape (1, h).
        bh (np.ndarray): Candidate bias, shape (1, h).
        by (np.ndarray): Output bias, shape (1, o).
    """

    def __init__(self, i, h, o):
        """
        Initialize weights and biases.

        Parameters:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # initializating Weights in order
        self.Wz = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wr = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wh = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wy = np.random.normal(size=(h, o))  # size = (15, 5)

        # initializating bias in order
        self.bz = np.zeros(shape=(1, h))
        self.br = np.zeros(shape=(1, h))
        self.bh = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
        Forward propagation for one time step.

        Parameters:
            h_prev (np.ndarray):
            Previous hidden state, shape (m, h).
            x_t (np.ndarray):
            Input data, shape (m, i).

        Returns:
            h_next (np.ndarray):
            Next hidden state, shape (m, h).
            y (np.ndarray): Output data, shape (m, o).
        """
        # https://victorzhou.com/blog/intro-to-rnns/

        x = np.concatenate((h_prev, x_t), axis=1)

        # gate z:
        z = np.dot(x, self.Wz) + self.bz
        # activating usng sigmoid
        z = 1 / (1 + np.exp(-z))

        # gate r:
        r = np.dot(x, self.Wr) + self.br
        # activating using sigmoid
        r = 1 / (1 + np.exp(-r))

        x = np.concatenate((r * h_prev, x_t), axis=1)
        h = np.tanh(np.dot(x, self.Wh) + self.bh)
        h_t = z * h + (1 - z) * h_prev

        # ŷ = Wₕᵧ · hₜ + bᵧ
        ŷ = np.dot(h_t, self.Wy) + self.by

        # Activating using softmax
        y = (np.exp(ŷ) / np.sum(np.exp(ŷ), axis=1, keepdims=True))

        return h_t, y
