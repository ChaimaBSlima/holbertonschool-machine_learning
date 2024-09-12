#!/usr/bin/env python3
""" Task 11 : 11. Learning Rate Decay """


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in NumPy.

    Args:
        alpha (float):
            The original learning rate.
        decay_rate (float):
            The rate at which the learning rate decays.
        global_step (int):
            The number of passes of gradient descent that have elapsed.
        decay_step (int):
            The number of passes of gradient descent that should occur
            before decaying the learning rate.

    Returns:
        float:
            The decayed learning rate.
    """
    α = alpha
    dr = decay_rate

    decayed_learning_rate = α / (1 + dr * int(global_step / decay_step))
    return (decayed_learning_rate)
