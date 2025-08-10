#!/usr/bin/env python3
""" Pandas function that convert a numpy array to a dataframe """
import pandas as pd


def from_numpy(array):
    """ Convert an np array to a dataframe

    Args:
        array (numpy array): input array

    Returns:
        pandas dataframe
    """
    return pd.DataFrame(array)
