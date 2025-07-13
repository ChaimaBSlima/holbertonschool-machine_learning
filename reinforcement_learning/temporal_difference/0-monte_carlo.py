#!/usr/bin/env python3
"""
Monte Carlo first-visit prediction algorithm.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Perform first-visit Monte Carlo prediction.

    Args:
        env: OpenAI Gym environment instance.
        V: numpy.ndarray of shape (s,) containing value estimates.
        policy: function(state) -> action.
        episodes: number of episodes to sample.
        max_steps: max steps per episode.
        alpha: learning rate.
        gamma: discount factor.

    Returns:
        Updated value function V.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            if done or truncated:
                break

        G = 0
        visited = set()
        for state, reward in reversed(episode):
            G = gamma * G + reward
            if state not in visited:
                visited.add(state)
                V[state] += alpha * (G - V[state])

    return V
