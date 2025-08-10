#!/usr/bin/env python3
"""
Module that contains a function to concatenate two DataFrames
"""
import pandas as pd


def concat(df1, df2):
    """
    Takes two pd.DataFrame objects and:
    - Indexes both dataframes on their Timestamp columns
    - Includes all timestamps from df2 (bitstamp) up
    to and including timestamp 1417411920
    - Concatenates the selected rows from df2 to the top of df1 (coinbase)
    - Adds keys to the concatenated data, labeling the rows
    from df2 as bitstamp and the rows from df1 as coinbase

    Args:
        df1: pd.DataFrame (coinbase data)
        df2: pd.DataFrame (bitstamp data)

    Returns:
        pd.DataFrame: the concatenated DataFrame
    """
    # Import index function
    index = __import__('10-index').index
    # Index both dataframes on their Timestamp columns
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Filter df2 to include all timestamps up to and including 1417411920
    df2_filtered = df2_indexed[df2_indexed.index <= 1417411920]

    # Concatenate the filtered df2 to the top of df1 with keys
    df_concatenated = pd.concat(
        [df2_filtered, df1_indexed], keys=["bitstamp", "coinbase"]
    )

    return df_concatenated
