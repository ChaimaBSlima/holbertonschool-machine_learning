#!/usr/bin/env python3
""" Task 1: 1. Pooling Forward Prop """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation for a pooling layer.

    Args:
    A_prev (numpy.ndarray):
        Input data of shape (m, h_prev, w_prev, c_prev) where:
        - m is the number of examples,
        - h_prev is the height of the previous layer,
        - w_prev is the width of the previous layer,
        - c_prev is the number of channels in the previous layer.

    kernel_shape (tuple):
        Size of the pooling kernel, as (kh, kw) where:
        - kh is the kernel height,
        - kw is the kernel width.

    stride (tuple, optional):
        Stride of the pooling operation, as (stride_height, stride_width).
        Default is (1, 1).

    mode (str, optional):
        Pooling mode to use, either 'max' for max pooling or 'avg'
        for average pooling.
        Default is 'max'.

    Returns:
    numpy.ndarray:
        Output of the pooling layer, of shape (m, n_h, n_w, c_prev) where:
        - n_h is the height of the output,
        - n_w is the width of the output,
        - c_prev is the number of channels (same as input channels).
    """

    # Retrieving dimensions from A_prev shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    # Retrieving dimensions from kerner_shape
    (kh, kw) = kernel_shape

    # Retrieving stride
    (sh, sw) = stride


    pw, ph = 0, 0

    # Compute the output dimensions
    n_h = int(((h_prev - kh) / sh) + 1)
    n_w = int(((w_prev - kw) / sw) + 1)

    # Initialize the output with zeros
    output = np.zeros((m, n_h, n_w, c_prev))

    # Looping over vertical and horizontal axis of output volume
    for x in range(n_w):
        for y in range(n_h):
            # element-wise multiplication of the kernel and the image
            if mode == 'max':
                output[:, y, x, :] = \
                    np.max(A_prev[:,
                                  y * sh: y * sh + kh,
                                  x * sw: x * sw + kw], axis=(1, 2))
            if mode == 'avg':
                output[:, y, x, :] = \
                    np.mean(A_prev[:,
                                   y * sh: y * sh + kh,
                                   x * sw: x * sw + kw], axis=(1, 2))

    return output
