#!/usr/bin/env python3
"""Program that computes to policy with a weight of a matrix"""
import numpy as np


def policy(matrix, weight):
    """Function that computes to policy with a weight of a matrix"""
    z = matrix.dot(weight)
    exp = np.exp(z)
    return exp / exp.sum()


def policy_gradient(state, weight):
    """Function that computes the Monte-Carlo policy
    gradient based on a state and a weight matrix"""
    P = policy(state, weight)
    action = np.random.choice(len(P[0]), p=P[0])
    s = P.reshape(-1, 1)
    softmax = np.diagflat(s) - np.dot(s, s.T)
    softmax = softmax[action, :]
    dlog = softmax / P[0, action]
    gradient = state.T.dot(dlog[None, :])
    return action, gradient
