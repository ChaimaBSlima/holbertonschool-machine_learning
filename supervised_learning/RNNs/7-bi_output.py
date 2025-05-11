#!/usr/bin/env python3
""" Task 7: 7. Bidirectional Output """

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
        by (ndarray): Bias for the output layer (shape: (1, o)).
    """

    def __init__(self, i, h, o):
        """
        Initialize the bidirectional RNN cell.

        Args:
            i (int): Dimensionality of the data input.
            h (int): Dimensionality of the hidden states.
            o (int): Dimensionality of the outputs.
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev (ndarray): Previous hidden state (shape: (m, h)).
            x_t (ndarray): Input data at time t (shape: (m, i)).

        Returns:
            h_next (ndarray): Next hidden state (shape: (m, h)).
        """
        x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(x, self.Whf) + self.bhf)
        return h_next

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
        h_prev = np.tanh(np.dot(x, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """
        Compute the outputs for the RNN using softmax.

        Args:
            H (ndarray): Concatenated hidden states from both directions
                         for all time steps (shape: (t, m, 2h)).

        Returns:
            Y (ndarray): Output predictions (shape: (t, m, o)).
        """
        t = H.shape[0]
        Y = []

        for i in range(t):
            y = np.dot(H[i], self.Wy) + self.by
            y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
            Y.append(y)

        return np.array(Y)
