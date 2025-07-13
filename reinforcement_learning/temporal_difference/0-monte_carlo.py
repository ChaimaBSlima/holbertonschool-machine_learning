#!/usr/bin/env python3
"""Monte Carlo algorithm implementation"""
import numpy as np


def episode_gen(env, policy, max_steps):
    """Generates an episode: list of (state, reward) pairs"""
    episode = []
    state = env.reset()
    for _ in range(max_steps):
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, reward))
        if done:
            break
        state = next_state
    return episode


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm using
    Every-Visit approach"""
    for ep in range(episodes):
        episode = episode_gen(env, policy, max_steps)
        for i, (state, _) in enumerate(episode):
            # Gt is return from time t
            Gt = sum(
                (gamma ** t) * reward for t, (_, reward)
                in enumerate(episode[i:])
            )
            # Every-Visit Monte Carlo update
            V[state] += alpha * (Gt - V[state])
    return V
