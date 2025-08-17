#!/usr/bin/env python3
"""Adjusts the contrast of an image using TensorFlow."""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image: A 3D tf.Tensor representing the input image to
        adjust the contrast.
        lower: A float, lower bound of the random contrast factor.
        upper: A float, upper bound of the random contrast factor.

    Returns:
        The contrast-adjusted image as a tf.Tensor.
    """
    return tf.image.random_contrast(image, lower=lower, upper=upper)
