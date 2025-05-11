#!/usr/bin/env python3
""" Task 0: 0. RNN Cell """
import numpy as np


class RNNCell:
    """
    Represents a single RNN cell.

    Attributes:
        Wh (np.ndarray):
        Weight matrix for hidden state and input, shape (i + h, h).
        Wy (np.ndarray):
        Weight matrix for hidden state to output, shape (h, o).
        bh (np.ndarray): Bias for hidden state, shape (1, h).
        by (np.ndarray): Bias for output, shape (1, o).
    """

    def __init__(self, i, h, o):
        """
        Initialize the RNN cell.

        Parameters:
        i (int): Dimensionality of the input data.
        h (int): Dimensionality of the hidden state.
        o (int): Dimensionality of the outputs.
        """
        # Weights for hidden state and input concatenated
        self.Wh = np.random.randn(i + h, h)
        # Weights for output
        self.Wy = np.random.randn(h, o)
        # Bias for hidden state
        self.bh = np.zeros((1, h))
        # Bias for output
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Parameters:
        h_prev (np.ndarray): Previous hidden state, shape (m, h).
        x_t (np.ndarray): Input data for the cell, shape (m, i).

        Returns:
        h_next (np.ndarray): Next hidden state, shape (m, h).
        y (np.ndarray): Output of the cell, shape (m, o).
        """
        # Concatenate previous hidden state and current input
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)

        # Compute output (before activation)
        y_linear = np.dot(h_next, self.Wy) + self.by

        # Apply softmax activation
        exp_shifted =\
            np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

        return h_next, y
