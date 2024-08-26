#!/usr/bin/env python3
""" Task 1: 1. Privatize Neuron """
import numpy as np


class Neuron:
    """
    Neuron class represents a single neuron in a neural network.

    Attributes:
    __W : numpy.ndarray
        The weight vector associated with the neuron.
    __b : float
        The bias term associated with the neuron. Initialized to 0.
    __A : float
        The activated output of the neuron. Initialized to 0.

    Methods:
    __init__(self, nx)
        Initializes the neuron with the given number of input features.
    W(self)
        Returns the weight vector of the neuron.
    b(self)
        Returns the bias term of the neuron.
    A(self)
        Returns the activated output of the neuron.
    """

    def __init__(self, nx):
        """
        Initializes the Neuron instance.

        Parameters:
        nx : int
            The number of input features to the neuron. Must be a positive
            integer.

        Raises:
        TypeError
            If `nx` is not an integer.
        ValueError
            If `nx` is less than 1.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.__W = np.random.normal(0, 1, (1, nx))
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        """
        Returns the weight vector of the neuron.

        Returns:
        numpy.ndarray
            The weight vector of the neuron.
        """
        return self.__W

    @property
    def b(self):
        """
        Returns the bias term of the neuron.

        Returns:
        float
            The bias term of the neuron.
        """
        return self.__b

    @property
    def A(self):
        """
        Returns the activated output of the neuron.

        Returns:
        float
            The activated output of the neuron.
        """
        return self.__A
