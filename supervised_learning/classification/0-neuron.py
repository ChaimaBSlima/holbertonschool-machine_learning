#!/usr/bin/env python3
""" Task 0: 0. Neuron"""
import numpy as np


class Neuron:
    """
    A class used to represent a single neuron in a neural network.

    Attributes
    W : numpy.ndarray
        The weight vector for the neuron
    b : float
        The bias for the neuron, initialized to 0.
    A : float
        The activation output of the neuron, initialized to 0.

    Methods
    __init__(self, nx)
        Initializes a neuron with `nx` input features.
    """
    def __init__(self, nx):
        """
        Initialize the Neuron instance.

        Parameters
        nx : int
            The number of input features to the neuron. It must be a
            positive integer.

        Raises
        TypeError
            If `nx` is not an integer.
        ValueError
            If `nx` is less than 1.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(0, 1, (1, nx))
        self.b = 0
        self.A = 0
