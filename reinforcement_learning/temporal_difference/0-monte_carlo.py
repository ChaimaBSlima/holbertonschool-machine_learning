#!/usr/bin/env python3
"""First-visit Monte Carlo algorithm"""


import numpy as np


def monte_carlo(
    env, V, policy, episodes=5000,
    max_steps=100, alpha=0.1, gamma=0.99
):
    """Estimates V-value with first-visit Monte Carlo algorithm"""
    for episode in range(episodes):
        state, _ = env.reset()
        episode_info = []
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_info.append((state, action, reward))
            if terminated or truncated:
                break
            state = next_state

        G = 0
        for ep_index in range(len(episode_info) - 1, - 1, - 1):
            state, action, reward = episode_info[ep_index]
            G = gamma * G + reward
            visited_states = [
                prev_state for prev_state, _, _ in episode_info[:episode]
            ]
            if state not in visited_states:
                V[state] = V[state] + alpha * (G - V[state])

    return V
