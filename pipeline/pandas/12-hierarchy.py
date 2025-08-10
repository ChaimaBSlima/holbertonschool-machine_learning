#!/usr/bin/env python3
"""
Module that contains a function to create a hierarchical
DataFrame with Timestamp as first level
"""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Takes two pd.DataFrame objects and:
    - Rearranges the MultiIndex so that Timestamp is the first level
    - Concatenates the bitstamp and coinbase tables from
    timestamps 1417411980 to 1417417980, inclusive
    - Adds keys to the data, labeling rows from df2 as
    bitstamp and rows from df1 as coinbase
    - Ensures the data is displayed in chronological order

    Args:
        df1: pd.DataFrame (coinbase data)
        df2: pd.DataFrame (bitstamp data)

    Returns:
        pd.DataFrame: the concatenated DataFrame with hierarchical index
    """
    # Index both dataframes on their Timestamp columns
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Filter both dataframes to include timestamps
    # from 1417411980 to 1417417980, inclusive
    df1_filtered = df1_indexed[
        (df1_indexed.index >= 1417411980) & (df1_indexed.index <= 1417417980)
    ]
    df2_filtered = df2_indexed[
        (df2_indexed.index >= 1417411980) & (df2_indexed.index <= 1417417980)
    ]

    # Concatenate with keys to label the data sources
    df_concatenated = pd.concat(
        [df2_filtered, df1_filtered], keys=["bitstamp", "coinbase"]
    )

    # Rearrange the MultiIndex so that Timestamp is the first level
    # Current structure: (exchange, timestamp) -> Want: (timestamp, exchange)
    df_swapped = df_concatenated.swaplevel(0, 1)

    # Sort by the index to ensure chronological order
    # (timestamp first, then exchange)
    df_sorted = df_swapped.sort_index()

    return df_sorted
