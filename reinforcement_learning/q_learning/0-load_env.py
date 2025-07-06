#!/usr/bin/env python3
""" Task 0: 0. Load the FrozenLake environment"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """[Function that loads the pre-made FrozenLakeEnv env from OpenAI’s gym]
    Args:
        desc ([list], optional):    [list of lists with a custom description
                                    of the map to load for the environment].
                                    Defaults to None.
        map_name ([str], optional): [string with the pre-made map to load].
                                Defaults to None.
        is_slippery (bool, optional): [description]. Defaults to False.
    Returns: the environment
    """

    # load all enviroments
    # load the very basic taxi environment.
    # env = gym.make("Taxi-v2")
    # To initialize the environment, we must reset it.
    # determine the total number of possible states:
    # env.observation_space.n
    # If you would like to visualize the current state, type the following:
    # env.render()

    env = gym.make("FrozenLake-v0",
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery,
                   render_mode="ansi")
    env.reset()
    return env
