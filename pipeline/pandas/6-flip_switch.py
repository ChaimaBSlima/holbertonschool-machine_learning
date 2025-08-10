#!/usr/bin/env python3
"""
Module that contains a function to flip and switch a DataFrame
"""


def flip_switch(df):
    """
    Takes a pd.DataFrame and:
    - Sorts the data in reverse chronological order
    - Transposes the sorted dataframe

    Args:
        df: pd.DataFrame to transform

    Returns:
        pd.DataFrame: the transformed DataFrame
    """
    # Sort the data in reverse chronological order (descending by index)
    # Since the data is already in chronological order, we reverse it
    df_sorted = df.sort_index(ascending=False)

    # Transpose the sorted dataframe
    df_transposed = df_sorted.T

    return df_transposed

