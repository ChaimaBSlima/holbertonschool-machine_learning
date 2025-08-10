#!/usr/bin/env python3
"""
Module that contains a function to sort a DataFrame by High price
"""


def high(df):
    """
    Takes a pd.DataFrame and:
    - Sorts it by the High price in descending order

    Args:
        df: pd.DataFrame to sort

    Returns:
        pd.DataFrame: the sorted DataFrame
    """
    # Sort by the High column in descending order
    df_sorted = df.sort_values(by="High", ascending=False)

    return df_sorted

