#!/usr/bin/env python3
""" Task 4: 4. Forward Propagation with Dropout """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Performs forward propagation with dropout regularization for
    a neural network.

    Parameters:
    X (numpy.ndarray): Input data, shape (features, examples).
    weights (dict):
    A dictionary containing the weights and biases of the network.
            - W1, W2, ..., WL are the weight matrices for each layer.
            - b1, b2, ..., bL are the bias vectors for each layer.
    L (int): The number of layers in the neural network.
    keep_prob (float): The probability of keeping a unit active during dropout.

    Returns:
    dict:
    A dictionary containing the activations (A) and dropout masks (D)
    at each layer:
          - A0, A1, ..., AL are the activations.
          - D1, D2, ..., DL are the dropout masks for each layer.
    """

    cache = {}
    cache['A0'] = X

    for la in range(1, L + 1):
        keyA = "A{}".format(la)
        keyA_p = "A{}".format(la - 1)
        keyD = "D{}".format(la)
        keyW = "W{}".format(la)
        keyb = "b{}".format(la)

        Z = np.matmul(weights[keyW], cache[keyA_p]) + weights[keyb]

        if la != L:
            A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            cache[keyD] = D
            A *= D
            A /= keep_prob
            cache[keyA] = A
        else:
            # Softmax
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
            cache[keyA] = A

    return cache
