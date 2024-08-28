#!/usr/bin/env python3
""" Task 28: 28. All the Activations """
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Defines a deep neural network for performing binary classification.

    Attributes:
        nx (int): Number of input features.
        layers (list): List representing the number of nodes
        in each layer of the network.
        L (int): Number of layers in the neural network.
        cache (dict): Dictionary to hold the intermediary
        values of the network (i.e., the activations).
        weights (dict): Dictionary to hold the weights
        and biases of the network.

    Methods:
        __init__(self, nx, layers)
            Initializes the deep neural network with given input features
            and nodes in each layer.
        L(self)
            Property getter for the number of layers in the network.
        cache(self)
            Property getter for the intermediary values in the network.
        weights(self)
            Property getter for the weights and biases of the network.
        sigmoid(self, z)
            Applies the sigmoid activation function.
        softmax(self, z)
            Applies the softmax activation function.
        forward_prop(self, X)
            Calculates the forward propagation of the neural network.
        evaluate(self, X, Y)
            Evaluates the neural network’s predictions.
        gradient_descent(self, Y, cache, alpha=0.05)
            Performs one pass of gradient descent on the neural network.
        train(self, X, Y, iterations=5000, alpha=0.05)
            Trains the deep neural network using forward propagation and
            gradient descent.
        save(self, filename)
            Save the instance object to a file in pickle format.
        load(filename)
            Load a pickled DeepNeuralNetwork object from a file.
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network.

        Args:
            nx (int): Number of input features.
            layers (list): List representing the number of nodes
            in each layer of the network.

        Raises:
            TypeError: If `nx` is not an integer.
            ValueError: If `nx` is less than 1.
            TypeError: If `layers` is not a list of positive integers.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__activation = activation
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            wkey = "W{}".format(i + 1)
            bkey = "b{}".format(i + 1)

            self.__weights[bkey] = np.zeros((layers[i], 1))

            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                w = np.random.randn(layers[i], layers[i - 1])
                w = w * np.sqrt(2 / layers[i - 1])
            self.__weights[wkey] = w

    @property
    def L(self):
        """ Property getter for the number of
        layers in the network."""
        return self.__L

    @property
    def cache(self):
        """ Property getter for the intermediary
        values in the network. """
        return self.__cache

    @property
    def weights(self):
        """ Property getter for the weights and
        biases of the network. """
        return self.__weights

    def sigmoid(self, z):
        """
        Applies the sigmoid activation function.

        Args:
            z (numpy.ndarray): A numpy array with shape
            (nx, m) that contains the input data.
                - nx (int): The number of input features
                to the neuron.
                - m (int): The number of examples.

        Returns:
            numpy.ndarray: The activated output using
            the sigmoid function.
        """
        y_hat = 1 / (1 + np.exp(-z))
        return y_hat

    def softmax(self, z):
        """
        Applies the softmax activation function.

        Args:
            z (numpy.ndarray): A numpy array with shape
            (nx, m) that contains the input data.
                - nx (int): The number of input features
                to the neuron.
                - m (int): The number of examples.

        Returns:
            numpy.ndarray: The activated output using
            softmax function.
        """
        y_hat = np.exp(z - np.max(z))
        return y_hat / y_hat.sum(axis=0)

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Array of shape (nx, m) containing
            the input data.
                - nx (int): Number of input features.
                - m (int): Number of examples.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The activations of the output layer
                (A_L) with shape (1, m).
                - dict: The cache dictionary containing the intermediary
                activations of each layer.
        """
        self.__cache['A0'] = X

        for i in range(self.__L):
            wkey = "W{}".format(i + 1)
            bkey = "b{}".format(i + 1)
            Aprevkey = "A{}".format(i)
            Akey = "A{}".format(i + 1)
            W = self.__weights[wkey]
            b = self.__weights[bkey]
            Aprev = self.__cache[Aprevkey]

            z = np.matmul(W, Aprev) + b
            if i < self.__L - 1:
                self.__cache[Akey] = self.sigmoid(z)
            else:
                self.__cache[Akey] = self.softmax(z)

        return (self.__cache[Akey], self.__cache)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): Array with shape (1, m)
            containing the correct labels for the input data.
                - m (int): Number of examples.
            A (numpy.ndarray): Array with shape (1, m)
            containing the activated output of the network for
            each example.

        Returns:
            float: The cost of the model, calculated using the
            logistic regression cost function.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.

        Args:
            X (numpy.ndarray):
            Array with shape (nx, m) containing the input data.
                - nx (int): Number of input features.
                - m (int): Number of examples.
            Y (numpy.ndarray):
            Array with shape (1, m) containing the correct labels for
            the input data.
                - m (int): Number of examples.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: Array with shape (1, m)
                representing the predicted labels (0 or 1) for each example.
                - float: The cost of the model, calculated
                using the logistic regression cost function.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        Y_hat = np.max(A, axis=0)
        A = np.where(A == Y_hat, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent on the neural network.

        Args:
            Y (numpy.ndarray):
            Array with shape (1, m) containing the correct labels for
            the input data.
                - m (int): Number of examples.
            cache (dict):
            Dictionary containing the intermediary values of the network
            during forward propagation.
                - Keys are strings in the format 'A{i}', where i is the
                layer index, and values are the activations.
            alpha (float): The learning rate. Must be a positive float.

        Updates:
            The method updates the weights and biases of the neural
            network in-place.

        Raises:
            TypeError: If `alpha` is not a float.
            ValueError: If `alpha` is less than or equal to 0.
        """
        m = Y.shape[1]
        weights_c = self.__weights.copy()
        for i in range(self.__L, 0, -1):
            A = cache["A" + str(i)]
            if i == self.__L:
                dz = A - Y
            else:
                if self.__activation == "sig":
                    g = A * (1 - A)
                    dz = (weights_c["W" + str(i + 1)].T @ dz) * g
                elif self.__activation == "tanh":
                    g = 1 - (A ** 2)
                    dz = (weights_c["W" + str(i + 1)].T @ dz) * g
            dw = (dz @ cache["A" + str(i - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights["W" + str(i)] = self.__weights[
                    "W" + str(i)] - (alpha * dw)
            self.__weights["b" + str(i)] = self.__weights[
                    "b" + str(i)] - (alpha * db)

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the deep neural network using forward
        propagation and gradient descent, with options for
        verbosity and visualization.

        Args:
            X (numpy.ndarray):
            Array with shape (nx, m) containing the input data.
                - nx (int): Number of input features.
                - m (int): Number of examples.
            Y (numpy.ndarray):
            Array with shape (1, m) containing the correct labels
            for the input data.
                - m (int): Number of examples.
            iterations (int, optional):
                Number of iterations to train over. Default is 5000.
            alpha (float, optional):
                Learning rate for gradient descent. Default is 0.05.
            verbose (bool, optional):
                If True, prints the cost after every `step` iterations.
                Default is True.
            graph (bool, optional):
                If True, plots and saves the training cost as a graph after
                training. Default is True.
            step (int, optional):
                The interval (in iterations) at which to print the cost and
                plot the graph. Default is 100.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: Array with shape (1, m) of the network's
                predicted labels after training.
                - float: The cost of the network after training.

        Raises:
            TypeError: If `iterations` is not an integer,
            `alpha` is not a float, or `step` is not an integer.
            ValueError: If `iterations` is less than or equal to 0,
            `alpha` is less than or equal to 0, or `step` is less
            than or equal to 0 and greater than `iterations`.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        cost_list = []
        steps_list = []
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, self.__cache['A{}'.format(self.L)])
                cost_list.append(cost)
                steps_list.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
        if graph is True:
            plt.plot(steps_list, cost_list, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.savefig("23-figure")
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Save the instance object to a file in pickle format.

        Args:
            filename (str): The name of the file to save the object to.
                            If the filename does not end with '.pkl',
                            the extension will be added automatically.

        Returns:
            None
        """
        try:
            pkl = ".pkl"
            if filename[-4:] != pkl:
                filename += pkl
            with open(filename, "wb") as f:
                pickle.dump(self,
                            f,
                            pickle.HIGHEST_PROTOCOL
                            )
        except Exception:
            pass

    @staticmethod
    def load(filename):
        """
        Load a pickled DeepNeuralNetwork object from a file.

        Args:
            filename (str): The name of the file to load the object from.

        Returns:
            DeepNeuralNetwork or None: The loaded object if successful;
                                    None if the file could not be loaded.
        """
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
