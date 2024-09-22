#!/usr/bin/env python3
""" Task 7: 7. Early Stopping """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines whether to stop training early based on the validation cost.

    Early stopping is a technique used to prevent overfitting by stopping the
    training process when the model's performance on the validation set no
    longer improves by a significant margin.

    Parameters:
    cost (float):
    The current cost (e.g., validation loss) of the model.
    opt_cost (float):
    The optimal (lowest) cost observed so far.
    threshold (float):
    The minimum change in cost improvement required to reset patience.
    patience (int):
    The maximum number of checks with no significant improvement allowed.
    count (int):
    The current count of consecutive checks with no significant improvement.

    Returns:
    tuple: A tuple containing:
           - stop (bool):
           Whether to stop training (True if patience is reached).
           - count (int):
           The updated count of consecutive checks with no improvement.
    """
    count = 0 if opt_cost - cost - threshold > 0 else count + 1
    stop = False
    if (count == patience):
        stop = True

    return (stop, count)
