#!/usr/bin/env python3
"""
Module that contains a function to fill missing values in a DataFrame
"""


def fill(df):
    """
    Takes a pd.DataFrame and:
    - Removes the Weighted_Price column
    - Fills missing values in the Close column with the previous row's value
    - Fills missing values in the High, Low, and Open columns
        with the corresponding Close value in the same row
    - Sets missing values in Volume_(BTC) and Volume_(Currency) to 0

    Args:
        df: pd.DataFrame to fill

    Returns:
        pd.DataFrame: the modified DataFrame with filled values
    """
    # Create a copy to avoid modifying the original DataFrame
    df_filled = df.copy()

    # Remove the Weighted_Price column
    df_filled = df_filled.drop(columns=["Weighted_Price"])

    # Fill missing values in the Close column with
    # the previous row's value (forward fill)
    df_filled["Close"] = df_filled["Close"].ffill()

    # Fill missing values in High, Low, and Open columns with
    # the corresponding Close value in the same row
    df_filled["High"] = df_filled["High"].fillna(df_filled["Close"])
    df_filled["Low"] = df_filled["Low"].fillna(df_filled["Close"])
    df_filled["Open"] = df_filled["Open"].fillna(df_filled["Close"])

    # Set missing values in Volume_(BTC) and Volume_(Currency) to 0
    df_filled["Volume_(BTC)"] = df_filled["Volume_(BTC)"].fillna(0)
    df_filled["Volume_(Currency)"] = df_filled["Volume_(Currency)"].fillna(0)

    return df_filled
