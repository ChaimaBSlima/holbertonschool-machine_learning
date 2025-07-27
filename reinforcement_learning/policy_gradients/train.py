#!/usr/bin/env python3
"""
Implement the training
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implements a full training.
    Args:
        env: the environment to train on
        nb_episodes: the number of episodes to train for
        alpha: the learning rate
        gamma: the discount factor
        show_result: whether to show the result of the training
    Returns:
        all values of the score (sum of all rewards during  one episode loop)
    """
    weights = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )

    scores = []

    for episode in range(nb_episodes):

        state = env.reset()[0]

        episode_rewards = []

        episode_gradients = []

        done = False

        while not done:
            action, gradient = policy_gradient(
                state,
                weights
            )

            if show_result and episode % 1000 == 0:
                env.render()

            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_rewards.append(reward)

            episode_gradients.append(gradient)

            state = next_state

            done = terminated or truncated

        score = sum(episode_rewards)

        scores.append(score)

        for i, gradient in enumerate(episode_gradients):
            reward = sum(
                R * gamma ** indx for indx, R in enumerate(episode_rewards[i:])
            )

            weights += alpha * reward * gradient

        print(f"Episode: {episode}, Score: {score}")

    return scores
