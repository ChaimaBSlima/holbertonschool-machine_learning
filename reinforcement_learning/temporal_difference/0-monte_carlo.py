#!/usr/bin/env python3
"""Monte Carlo algorithm implementation"""
import numpy as np


def episode_gen(env, policy, max_steps):
    """Generates an episode as a list of (state, reward) tuples"""
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
    """Performs the Monte Carlo algorithm using Every-Visit approach"""
    for _ in range(episodes):
        episode = episode_gen(env, policy, max_steps)
        for i, (state, _) in enumerate(episode):
            # Compute Gt = sum of discounted future rewards
            Gt = 0
            for t, (_, reward) in enumerate(episode[i:]):
                Gt += (gamma ** t) * reward
            # Update V[state] using alpha and current estimate
            V[state] += alpha * (Gt - V[state])
    return V
