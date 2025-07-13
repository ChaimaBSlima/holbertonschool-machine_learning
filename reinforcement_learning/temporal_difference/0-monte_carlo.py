#!/usr/bin/env python3
"""Performs the Monte Carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Monte Carlo every-visit algorithm"""

    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)  # Uses the main's policy(), which depends on np.random
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, r = episode[t]
            G = r + gamma * G
            if s not in visited:
                visited.add(s)
                V[s] += alpha * (G - V[s])

    return V
