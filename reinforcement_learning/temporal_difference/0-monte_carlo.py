#!/usr/bin/env python3
"""Monte Carlo algorithm implementation"""
import numpy as np


def episode_gen(env, policy, max_steps):
    """Generates an episode: list of (state, reward) pairs"""
    episode = []
    state, _ = env.reset()
    for _ in range(max_steps):
        action = policy(state)
        next_state, reward, done, truncated, _ = env.step(action)
        episode.append((state, reward))
        if done or truncated:
            break
        state = next_state
    return episode


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """Monte Carlo every-visit value estimation"""
    for _ in range(episodes):
        episode = episode_gen(env, policy, max_steps)

        # Compute returns for each state
        G = 0
        visited = set()
        for i in reversed(range(len(episode))):
            state, reward = episode[i]
            G = reward + gamma * G

            if state in visited:
                continue
            visited.add(state)

            if V[state] == -1:  # skip holes
                continue

            V[state] += alpha * (G - V[state])

    return V
