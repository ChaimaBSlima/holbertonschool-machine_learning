#!/usr/bin/env python3
"""Monte Carlo algorithm implementation"""
import numpy as np


def episode_gen(env, policy, max_steps):
    """Generates an episode as list of (state, reward)"""
    episode = []
    visited = set()
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
    """Performs the Monte Carlo algorithm with first-visit updates"""
    for _ in range(episodes):
        episode = episode_gen(env, policy, max_steps)
        visited = set()
        G = 0
        for i in reversed(range(len(episode))):
            state, reward = episode[i]
            G = reward + gamma * G
            if state not in visited:
                visited.add(state)
                if V[state] == -1:  # Don't update hole states
                    continue
                V[state] += alpha * (G - V[state])
    return V
