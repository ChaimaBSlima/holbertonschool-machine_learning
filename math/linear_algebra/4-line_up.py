#!/usr/bin/env python3
"""
Task 4: 4. Line Up
"""


def add_arrays(arr1, arr2):
    """
     Add two arrays element-wise.

    Args:
        arr1: the first array.
        arr2: the second array.

    Returns:
        None: if arr1 and arr2 are not the same shape.
        SommedArr: the new array with sommed values.
    """
    if len(arr1) != len(arr2):
        return (None)
    SommedArr = []
    for i in range(len(arr1)):
        SommedArr.append(arr1[i] + arr2[i])
    return SommedArr
