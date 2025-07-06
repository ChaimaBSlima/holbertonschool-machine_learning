#!/usr/bin/env python3
""" Task 2: 2. Epsilon Greedy """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
        Selects the next action using the epsilon-greedy policy.
    
    Args:
        Q (numpy.ndarray): Q-table of shape (states, actions)
        state (int): current state
        epsilon (float): probability of choosing a random action (exploration)
    
    Returns:
        int: index of the action selected
    """

    # https://www.youtube.com/watch?v=HGeI30uATws&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=9
    # ε = epsilon
    # print(Q)
    # print(Q.shape)
    # print(Q[state])
    # print (state, epsilon)

    if np.random.uniform(0, 1) < epsilon:
        return int(np.random.randint(Q.shape[1]))
    else:
        return int(np.argmax(Q[state]))
