#!/usr/bin/env python3
"""Monte Carlo algorithm implementation"""
import numpy as np


def episode_gen(env, policy, max_steps):
    """Generates an episode as a list of (state, reward) pairs"""
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
    """Monte Carlo every-visit value estimation with alpha update"""
    for _ in range(episodes):
        episode = episode_gen(env, policy, max_steps)
        for i in range(len(episode)):
            state, _ = episode[i]
            if V[state] == -1:  # Don't update holes
                continue
            G = 0
            for j, (_, reward) in enumerate(episode[i:]):
                G += (gamma ** j) * reward
            V[state] += alpha * (G - V[state])
    return V
