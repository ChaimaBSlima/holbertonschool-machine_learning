#!/usr/bin/env python3
"""
Module that contains a function to slice a DataFrame
"""


def slice(df):
    """
    Takes a pd.DataFrame and:
    - Extracts the columns High, Low, Close, and Volume_(BTC)
    - Selects every 60th row from these columns

    Args:
        df: pd.DataFrame to slice

    Returns:
        pd.DataFrame: the sliced DataFrame
    """
    # Extract the specified columns
    columns_to_extract = ["High", "Low", "Close", "Volume_(BTC)"]
    df_extracted = df[columns_to_extract]

    # Select every 60th row (starting from index 0, then 60, 120, etc.)
    df_sliced = df_extracted.iloc[::60]

    return df_sliced
