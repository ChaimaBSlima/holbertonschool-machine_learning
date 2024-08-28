#!/usr/bin/env python3
""" Deep Neural Network """
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers, activation='sig'):
        """Instantiation Method

        Args:
            nx: number of input features
            layers: list representing the number of nodes in each
                layer of the network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) != list or len(layers) == 0:
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
        """ property setter for the attribute """
        return self.__L

    @property
    def cache(self):
        """ property setter for the attribute """
        return self.__cache

    @property
    def weights(self):
        """ property setter for the attribute """
        return self.__weights

    def sigmoid(self, z):
        """
        Applies the sigmoid activation function
        Arguments:
        - z (numpy.ndattay): with shape (nx, m) that contains the input data
         * nx is the number of input features to the neuron.
         * m is the number of examples
        Updates the private attribute __A
        The neuron should use a sigmoid activation function
        Return:
        The private attribute A
        """
        y_hat = 1 / (1 + np.exp(-z))
        return y_hat

    def softmax(self, z):
        """
        Applies the softmax activation function
        Arguments:
        - z (numpy.ndattay): with shape (nx, m) that contains the input data
         * nx is the number of input features to the neuron.
         * m is the number of examples
        Updates the private attribute __A
        The neuron should use a sigmoid activation function
        Return:
        The private attribute y_hat
        """
        y_hat = np.exp(z - np.max(z))
        return y_hat / y_hat.sum(axis=0)

    def forward_prop(self, X):
        """Calculates the forward propagation of the
            neural network

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
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
                if self.__activation == 'sig':
                    self.__cache[Akey] = self.sigmoid(z)
                else:
                    self.__cache[Akey] = np.tanh(z)
            else:
                self.__cache[Akey] = self.softmax(z)

        return (self.__cache[Akey], self.__cache)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) that contains the
                correct labels for the input data
            A: numpy.ndarray with shape (1, m) containing the activated
                output of the neuron for each example
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m

        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        Y_hat = np.max(A, axis=0)
        A = np.where(A == Y_hat, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network

        Args:
            Y: numpy.ndarray with shape (1, m) that contains the
                correct labels for the input data
            cache: dictionary containing all the intermediary
                values of the network
            alpha: the learning rate
        """
        m = Y.shape[1]
        # dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
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
            # dz for next iteration
            self.__weights["W" + str(i)] = self.__weights[
                    "W" + str(i)] - (alpha * dw)
            self.__weights["b" + str(i)] = self.__weights[
                    "b" + str(i)] - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct
              labels for the input data
            iterations: number of iterations to train over
            alpha: learning rate
            verbose: is a boolean that defines whether or not to print
              information about the training
            graph:  boolean that defines whether or not to graph information
              about the training once the training has completed.
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
            Save the instance object
            to a file in pickle format
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
            Loads a pickled
            DeepNeuralNetwork object
        """
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
