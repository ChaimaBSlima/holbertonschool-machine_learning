#!/usr/bin/env python3
"""
Module that contains a function to prune a DataFrame by removing NaN values
"""


def prune(df):
    """
    Takes a pd.DataFrame and:
    - Removes any entries where Close has NaN values

    Args:
        df: pd.DataFrame to prune

    Returns:
        pd.DataFrame: the modified DataFrame with NaN Close values removed
    """
    # Remove rows where Close column has NaN values
    df_pruned = df.dropna(subset=["Close"])

    return df_pruned
