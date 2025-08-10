#!/usr/bin/env python3
"""
Module that contains a function to rename and process a DataFrame
"""

import pandas as pd


def rename(df):
    """
    Takes a pd.DataFrame as input and performs the following:
    - Renames the Timestamp column to Datetime
    - Converts the timestamp values to datetime values
    - Displays only the Datetime and Close columns

    Args:
        df: pd.DataFrame containing a column named Timestamp

    Returns:
        pd.DataFrame: the modified DataFrame with Datetime and Close columns
    """
    # Create a copy to avoid modifying the original DataFrame
    df_modified = df.copy()

    # Rename the Timestamp column to Datetime
    df_modified = df_modified.rename(columns={"Timestamp": "Datetime"})

    # Convert the timestamp values to datetime values
    # The timestamps appear to be Unix timestamps (seconds since epoch)
    df_modified["Datetime"] = pd.to_datetime(df_modified["Datetime"], unit="s")

    # Return only the Datetime and Close columns
    return df_modified[["Datetime", "Close"]]
