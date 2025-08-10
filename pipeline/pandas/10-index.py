#!/usr/bin/env python3
"""
Module that contains a function to set the Timestamp column as the index
"""


def index(df):
    """
    Takes a pd.DataFrame and:
    - Sets the Timestamp column as the index of the dataframe

    Args:
        df: pd.DataFrame to modify

    Returns:
        pd.DataFrame: the modified DataFrame with Timestamp as index
    """
    # Set the Timestamp column as the index
    df_indexed = df.set_index("Timestamp")

    return df_indexed
