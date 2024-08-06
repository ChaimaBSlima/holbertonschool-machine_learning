#!/usr/bin/env python3
"""Task 0: 0. Line Graph"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Generate and display a line plot of the cubes of numbers from 0 to 10.

    Args:
        None

    Returns:
        None
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(np.arange(0, 11), y, 'r')
    plt.xlim((0, 10))
    plt.show()
