#!/usr/bin/env python3
""" Task 4: 4. Play """
import numpy as np


def play(env, Q, max_steps=100):
    """[Function that has the trained agent play an episode]
    Args:
        env ([instance]):   [FrozenLakeEnv instance]
        Q ([ndarray]):      [is a numpy.ndarray containing the Q-table]
        max_steps (int, optional): [maximum number of steps in the episode].
                                    Defaults to 100.
    Each state of the board should be displayed via the console
    You should always exploit the Q-table
    Returns: the total rewards for the episode
    """
    # Reset environment and get initial state
    state = env.reset()
    total_reward = 0
    renders = [env.render()]  # capture first render
    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        total_reward += reward
        renders.append(env.render())
        if done:
            break
    return total_reward, renders
