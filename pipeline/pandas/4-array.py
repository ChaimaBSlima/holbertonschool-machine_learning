#!/usr/bin/env python3
"""
Module that contains a function to convert DataFrame columns to numpy array
"""


def array(df):
    """
    Takes a pd.DataFrame as input and performs the following:
    - Selects the last 10 rows of the High and Close columns
    - Converts these selected values into a numpy.ndarray

    Args:
        df: pd.DataFrame containing columns named High and Close

    Returns:
        numpy.ndarray: the last 10 rows of High and Close columns
    """
    # Select the last 10 rows of High and Close columns
    last_10_rows = df[["High", "Close"]].tail(10)

    # Convert to numpy array
    return last_10_rows.to_numpy()
