#!/usr/bin/env python3
""" Task 0: 0. Valid Convolution """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Perform a valid convolution on grayscale images.

    Parameters:
    images (numpy.ndarray):
    A numpy array of shape (m, h, w) containing m grayscale images,
    each of height h and width w.
    kernel (numpy.ndarray):
    A numpy array of shape (kh, kw) representing the convolution kernel,
    where kh is the height and kw is the width of the kernel.

    Returns:
    numpy.ndarray:
    A numpy array of shape (m, H, W) containing the result of the convolution,
    where H is the output height and W is the output width. The output size is
    determined by the formula: H = h - kh + 1, W = w - kw + 1.
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    H = int(h - kh + 1)
    W = int(w - kw + 1)
    image_filter = np.zeros((m, H, W))
    for i in range(W):
        for j in range(H):
            image_filter[:, j, i] = (kernel
                                     * images[:, j: j + kh,
                                              i: i + kw]).sum(axis=(1, 2))
    return image_filter
