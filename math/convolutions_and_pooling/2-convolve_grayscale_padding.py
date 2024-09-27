#!/usr/bin/env python3
""" Task 2: 2. Convolution with Padding """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Perform a convolution on grayscale images with specified padding.

    Parameters:
    images (numpy.ndarray): A numpy array of shape (m, h, w) containing m grayscale images
                            of height h and width w.
    kernel (numpy.ndarray): A numpy array of shape (kh, kw) representing the convolution kernel,
                            where kh is the height and kw is the width of the kernel.
    padding (tuple): A tuple of (ph, pw) where ph is the height padding and pw is the width padding.

    Returns:
    numpy.ndarray: A numpy array of shape (m, H, W) containing the result of the convolution,
                   where H is the output height and W is the output width.
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph = padding[0]
    pw = padding[1]
    padded_image = np.pad(images,
                          pad_width=((0, 0), (ph, ph), (pw, pw)),
                          mode='constant', constant_values=0)
    H = int(padded_image.shape[1] - kh + 1)
    W = int(padded_image.shape[2] - kw + 1)
    image_filter = np.zeros((m, H, W))
    for i in range(W):
        for j in range(H):
            image_filter[:, j, i] = (kernel *
                                     padded_image[:, j: j + kh,
                                                  i: i + kw]).\
                                                  sum(axis=(1, 2))
    return image_filter
