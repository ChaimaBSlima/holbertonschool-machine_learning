#!/usr/bin/env python3
"""
Simple Policy function
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def policy(state, weight):
    """
    Compute the policy for a given state and weight.
    Args:
        state: The input state.
        weight: The weights of the policy.
    Returns:
        The action probabilities.
    """
    z = np.matmul(state, weight)
    return softmax(z)


def policy_gradient(state, weight):
    """
    Compute the Monte-Carlo policy gradient based on state and a weight matrix
    Args:
        state: matrix representing the current observation of the environment
        weight:  matrix of random weight
    Returns:
        The action and the gradient (in this order)
    """
    probabilities = policy(state, weight)

    action = np.random.choice(len(probabilities), p=probabilities)

    one_hot = np.zeros_like(probabilities)

    one_hot[action] = 1

    diff = one_hot - probabilities

    grad = np.outer(state, diff)

    return action, grad
