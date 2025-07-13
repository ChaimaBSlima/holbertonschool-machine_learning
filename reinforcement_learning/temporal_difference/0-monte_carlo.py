#!/usr/bin/env python3
"""Monte Carlo algorithm implementation"""
import numpy as np


def episode_gen(env, policy, max_steps):
    """Generates an episode as list of (state, reward) pairs"""
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


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm (every-visit)"""
    terminal_states = set()
    desc = env.unwrapped.desc.reshape(-1)

    for idx, val in enumerate(desc):
        if val in (b'H', b'G'):
            terminal_states.add(idx)

    for _ in range(episodes):
        episode = episode_gen(env, policy, max_steps)
        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            state, reward = episode[t]
            G = reward + gamma * G
            if state in visited or state in terminal_states:
                continue
            visited.add(state)
            V[state] += alpha * (G - V[state])
    return V
