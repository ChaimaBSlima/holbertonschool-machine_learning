#!/usr/bin/env python3
""" Task 6: 6. Bidirectional Cell Backward """

import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional recurrent cell for a simple RNN.

    Attributes:
        Whf (ndarray): Weights for the forward direction (shape: (i + h, h)).
        Whb (ndarray): Weights for the backward direction (shape: (i + h, h)).
        Wy (ndarray): Weights for the output layer (shape: (2h, o)).
        bhf (ndarray): Bias for the forward direction (shape: (1, h)).
        bhb (ndarray): Bias for the backward direction (shape: (1, h)).
        by (ndarray): Bias for the output (shape: (1, o)).
    """

    def __init__(self, i, h, o):
        """
        Initialize the bidirectional RNN cell.

        Args:
            i (int): Dimensionality of the data input.
            h (int): Dimensionality of the hidden states.
            o (int): Dimensionality of the outputs.
        """

        # initializating Weights in order
        self.Whf = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Whb = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wy = np.random.normal(size=(h + h, o))  # size = (30, 5)

        # initializating bias in order
        self.bhf = np.zeros(shape=(1, h))
        self.bhb = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev (ndarray): Previous hidden state (shape: (m, h)).
            x_t (ndarray): Input data at time t (shape: (m, i)).

        Returns:
            h_next (ndarray): Next hidden state (shape: (m, h)).
        """

        # https://victorzhou.com/blog/intro-to-rnns/

        x = np.concatenate((h_prev, x_t), axis=1)
        h_t = np.tanh(np.dot(x, self.Whf) + self.bhf)

        return h_t

    def backward(self, h_next, x_t):
        """
        Perform backward propagation for one time step.

        Args:
            h_next (ndarray): Next hidden state (shape: (m, h)).
            x_t (ndarray): Input data at time t (shape: (m, i)).

        Returns:
            h_prev (ndarray): Previous hidden state (shape: (m, h)).
        """

        x = np.concatenate((h_next, x_t), axis=1)
        h_pev = np.tanh(np.dot(x, self.Whb) + self.bhb)

        return h_pev
