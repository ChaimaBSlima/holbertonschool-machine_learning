#!/usr/bin/env python3
""" Task 0: 0. Placeholders"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_placeholders(nx, classes):
    """
    Creates and returns TensorFlow placeholders for input data and labels.

    Parameters:
    nx : int
        The number of feature columns in the input data
        (number of input features).
    classes : int
        The number of classes in the output labels
        (number of classes for classification).

    Returns:
    x : tf.placeholder
        A placeholder for the input data with shape
        [None, nx], where 'None' allows for a variable batch size.
    y : tf.placeholder
        A
    """
    x = tf.placeholder(float, shape=[None, nx], name='x')
    y = tf.placeholder(float, shape=[None, classes], name='y')

    return (x, y)
