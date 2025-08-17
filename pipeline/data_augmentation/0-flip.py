#!/usr/bin/env python3
"""Program that flips an image horizontally"""
import tensorflow as tf


def flip_image(image):
    """Function that flips an image horizontally"""
    return tf.image.flip_left_right(image)
