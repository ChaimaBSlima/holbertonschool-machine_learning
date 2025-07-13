#!/usr/bin/env python3
"""
Monte Carlo method for policy evaluation.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to update the value estimate.

    Args:
        env: The environment instance.
        V (numpy.ndarray): Value estimate array of shape (s,).
        policy (function): Function that takes a state and
        returns an action.
        episodes (int): Number of episodes to train over.
        max_steps (int): Maximum steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount rate.

    Returns:
        numpy.ndarray: Updated value estimate array.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []
        done = False
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated
            if done:
                break

        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            state_t, _, reward_t = episode[t]
            G = gamma * G + reward_t
            V[state_t] += alpha * (G - V[state_t])

    return V
