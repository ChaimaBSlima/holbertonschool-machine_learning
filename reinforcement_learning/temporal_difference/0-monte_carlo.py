#!/usr/bin/env python3
"""Monte Carlo algorithm implementation"""
import numpy as np


def episode_gen(env, policy, max_steps):
    """Generates an episode as list of (state, reward) tuples"""
    episode = []
    state, _ = env.reset()
    for _ in range(max_steps):
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, reward))
        if terminated or truncated:
            break
        state = next_state
    return episode


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """Performs Monte Carlo every-visit algorithm updating V in-place"""
    for _ in range(episodes):
        episode = episode_gen(env, policy, max_steps)

        G = 0
        # Walk backward through the episode to compute returns
        for t in reversed(range(len(episode))):
            state, reward = episode[t]
            G = reward + gamma * G

            # Skip holes (V[state] is initially -1) or goal (1.0)
            if V[state] in (-1.0, 1.0):
                continue

            # Every-visit MC: update every occurrence
            V[state] += alpha * (G - V[state])

    return V
