#!/usr/bin/env python3
""" Task 3: 3. Pooling Back Prop """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs the backward pass for a pooling layer.

    Args:
    dA (numpy.ndarray):
        Gradient of the cost with respect to the output of
        the pooling layer (post-pooling),
        of shape (m, h_new, w_new, c_new), where:
        - m is the number of examples,
        - h_new is the height of the output,
        - w_new is the width of the output,
        - c_new is the number of channels.

    A_prev (numpy.ndarray):
        Input from the previous layer (before pooling),
        of shape (m, h_prev, w_prev, c_prev), where:
        - m is the number of examples,
        - h_prev is the height of the input,
        - w_prev is the width of the input,
        - c_prev is the number of channels.

    kernel_shape (tuple):
        Size of the pooling window (kh, kw), where:
        - kh is the kernel height,
        - kw is the kernel width.

    stride (tuple, optional):
        Stride of the pooling, as (stride_height, stride_width).
        Default is (1, 1).

    mode (str, optional):
        The type of pooling operation. Either 'max' for max-
        or 'avg' for average-pooling.
        Default is 'max'.

    Returns:
    dA_prev (numpy.ndarray):
        Gradient of the cost with respect to the input of the pooling layer
        (A_prev),of the same shape as A_prev (m, h_prev, w_prev, c_prev).
    """

    # Retrieving dimensions from dA
    m, h_new, w_new, c_new = dA.shape

    # Retrieving dimensions from A_prev shape
    m, h_prev, w_prev, c_prev = A_prev.shape

    # Retrieving dimensions from kernel_shape
    kh, kw = kernel_shape

    # Retrieving stride
    sh, sw = stride

    # Initialize the output with zeros
    dA_prev = np.zeros_like(A_prev, dtype=dA.dtype)

    # Looping over vertical(h) and horizontal(w) axis of output volume
    for z in range(m):
        for y in range(h_new):
            for x in range(w_new):
                for v in range(c_new):
                    pool = A_prev[z, y * sh:(kh+y*sh), x * sw:(kw+x*sw), v]
                    dA_aux = dA[z, y, x, v]
                    if mode == 'max':
                        z_mask = np.zeros(kernel_shape)
                        _max = np.amax(pool)
                        np.place(z_mask, pool == _max, 1)
                        dA_prev[z, y * sh:(kh + y * sh),
                                x * sw:(kw+x*sw), v] += z_mask * dA_aux
                    if mode == 'avg':
                        avg = dA_aux / kh / kw
                        o_mask = np.ones(kernel_shape)
                        dA_prev[z, y * sh:(kh + y * sh),
                                x * sw:(kw+x*sw), v] += o_mask * avg
    return dA_prev
