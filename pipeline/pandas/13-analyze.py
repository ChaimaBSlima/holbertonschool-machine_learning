#!/usr/bin/env python3
"""
Module that contains a function to compute
descriptive statistics for a DataFrame
"""


def analyze(df):
    """
    Takes a pd.DataFrame and:
    - Computes descriptive statistics for all
    columns except the Timestamp column

    Args:
        df: pd.DataFrame to analyze

    Returns:
        pd.DataFrame: a new DataFrame containing descriptive statistics
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Remove the Timestamp column if it exists
    if "Timestamp" in df_copy.columns:
        df_copy = df_copy.drop(columns=["Timestamp"])

    # Compute descriptive statistics for all remaining columns
    stats = df_copy.describe()

    return stats
