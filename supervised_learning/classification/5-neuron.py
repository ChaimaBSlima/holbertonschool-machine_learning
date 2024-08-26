#!/usr/bin/env python3
""" Task 5: 5. Neuron Gradient Descent """
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
    forward_prop(self, X)
        Computes the forward propagation of the neuron.
    forward_prop(self, X)
        Calculates the cost of the model using logistic regression.
    evaluate(self, X, Y)
        Evaluates the neuron’s predictions.
    gradient_descent(self, X, Y, A, alpha=0.05)
        Performs one pass of gradient descent on the neuron.
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

    def forward_prop(self, X):
        """
        Computes the forward propagation of the neuron.

        Parameters:
        X : numpy.ndarray
            Input data, where each column represents a different
            example and each row represents a feature.

        Returns:
        numpy.ndarray
            The activated output of the neuron after applying the
            sigmoid function.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1.0 / (1.0 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Parameters:
        Y : numpy.ndarray
            A numpy array with shape (1, m) that contains the correct labels
            for the input data, where `m` is the number of examples.
        A : numpy.ndarray
            A numpy array with shape (1, m) that contains the activated
            output of the neuron for each example.

        Returns:
        float
            The cost of the model, computed using the logistic regression
            cost function.
        """
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions.

        Parameters:
        X : numpy.ndarray
            A numpy array with shape (nx, m) that contains the input data,
            where `nx` is the number of input features to the neuron and
            `m` is the number of examples.
        Y : numpy.ndarray
            A numpy array with shape (1, m) that contains the correct labels
            for the input data.

        Returns:
        A : numpy.ndarray
            A numpy array with shape (1, m)
        cost : float
            The cost of the model’s predictions
        """
        self.forward_prop(X)
        A = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one pass of gradient descent on the neuron.

        Parameters:
        X : numpy.ndarray
            A numpy array with shape (nx, m) that contains the input data,
            where `nx` is the number of input features to the neuron and
            `m` is the number of examples.
        Y : numpy.ndarray
            A numpy array with shape (1, m) that contains the correct labels
            for the input data.
        A : numpy.ndarray
            A numpy array with shape (1, m) that contains the activated
            output of the neuron for each example.
        alpha : float, optional
            The learning rate, which controls the step size in gradient
            descent. Defaults to 0.05.

        Returns:
            None
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(X, dZ.T) / m
        db = np.sum(dZ) / m
        self.__W = self.__W - (alpha * dW).T
        self.__b = self.__b - alpha * db
