#!/usr/bin/env python3
""" Task 3: 3. LSTM Cell """
import numpy as np


class LSTMCell:
    """
    Long Short-Term Memory (LSTM) cell.

    Attributes:
        Wf (np.ndarray): Forget gate weight, shape (i + h, h).
        Wu (np.ndarray): Update gate weight, shape (i + h, h).
        Wc (np.ndarray): Candidate cell weight, shape (i + h, h).
        Wo (np.ndarray): Output gate weight, shape (i + h, h).
        Wy (np.ndarray): Output weight, shape (h, o).
        bf (np.ndarray): Forget gate bias, shape (1, h).
        bu (np.ndarray): Update gate bias, shape (1, h).
        bc (np.ndarray): Candidate bias, shape (1, h).
        bo (np.ndarray): Output gate bias, shape (1, h).
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
        self.Wf = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wu = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wc = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wo = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wy = np.random.normal(size=(h, o))  # size = (15, 5)

        # initializating bias in order
        self.bf = np.zeros(shape=(1, h))
        self.bu = np.zeros(shape=(1, h))
        self.bc = np.zeros(shape=(1, h))
        self.bo = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Forward propagation for one time step.

        Parameters:
            h_prev (np.ndarray): Previous hidden state, shape (m, h).
            c_prev (np.ndarray): Previous cell state, shape (m, h).
            x_t (np.ndarray): Input data, shape (m, i).

        Returns:
            h_next (np.ndarray): Next hidden state, shape (m, h).
            c_next (np.ndarray): Next cell state, shape (m, h).
            y (np.ndarray): Output data, shape (m, o).
        """

        # https://victorzhou.com/blog/intro-to-rnns/

        x = np.concatenate((h_prev, x_t), axis=1)

        # gate u:
        u = np.dot(x, self.Wu) + self.bu
        # activating usng sigmoid
        u = 1 / (1 + np.exp(-u))

        # gate f:
        f = np.dot(x, self.Wf) + self.bf
        # activating usng sigmoid
        f = 1 / (1 + np.exp(-f))

        # gate o:
        o = np.dot(x, self.Wo) + self.bo
        # activating using sigmoid
        o = 1 / (1 + np.exp(-o))

        c = np.tanh(np.dot(x, self.Wc) + self.bc)
        c_t = u * c + f * c_prev
        h_t = o * np.tanh(c_t)

        # ŷ = Wₕᵧ · hₜ + bᵧ
        ŷ = np.dot(h_t, self.Wy) + self.by

        # Activating using softmax
        y = (np.exp(ŷ) / np.sum(np.exp(ŷ), axis=1, keepdims=True))

        return h_t, c_t, y
