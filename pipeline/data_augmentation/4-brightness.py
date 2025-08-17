#!/usr/bin/env python3
"""Program that randomly changes the brightness of an image"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Function that randomly changes the brightness of an image"""
    return tf.image.random_brightness(image, max_delta=max_delta)
