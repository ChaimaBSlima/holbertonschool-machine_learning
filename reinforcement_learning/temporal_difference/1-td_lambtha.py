#!/usr/bin/env python3
"""Temporal difference algorithm with trace decay parameter"""


import numpy as np


def td_lambtha(
    env, V, policy, lambtha, episodes=5000,
    max_steps=100, alpha=0.1, gamma=0.99
):
    """Updates V value based on TD with trace decay algorithm"""
    for episode in range(episodes):
        E = np.zeros(env.observation_space.n)
        state, _ = env.reset()
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            delta = reward + gamma * V[next_state] - V[state]
            E[state] += 1
            V += alpha * delta * E
            E *= gamma * lambtha
            state = next_state
            if terminated or truncated:
                break
    return V
