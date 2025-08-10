#!/usr/bin/env python3
"""Program that created a pd.DataFrame from a file"""
import pandas as pd


def from_file(filename, delimiter):
    """Reads a file and returns a dataframe

    Args:
        filename (csv): csv file containing the data to be read
        delimiter (str): delimeter of data

    Returns:
        pandas dataframe
    """
    return pd.read_csv(filename, delimiter=delimiter)
