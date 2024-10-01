#!/usr/bin/env python3
""" Task 2: 2. Convolutional Back Prop """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backward propagation for a convolutional layer.

    Args:
    dZ (numpy.ndarray):
        Gradient of the cost with respect to the output of the
        convolutional layer (post-activation),
        of shape (m, h_new, w_new, c_new) where:
        - m is the number of examples,
        - h_new is the height of the output,
        - w_new is the width of the output,
        - c_new is the number of filters (output channels).

    A_prev (numpy.ndarray):
        Output from the previous layer (input to the convolutional layer),
        of shape (m, h_prev, w_prev, c_prev) where:
        - m is the number of examples,
        - h_prev is the height of the previous layer,
        - w_prev is the width of the previous layer,
        - c_prev is the number of channels in the previous layer.

    W (numpy.ndarray):
        Weights (filters) of the convolutional layer,
        of shape (kh, kw, c_prev, c_new) where:
        - kh is the kernel height,
        - kw is the kernel width,
        - c_prev is the number of channels in the previous layer,
        - c_new is the number of filters.

    b (numpy.ndarray):
        Biases for the convolutional layer, of shape (1, 1, 1, c_new).

    padding (str, optional):
        Padding type, either 'same' or 'valid'.
        Default is 'same'.

    stride (tuple, optional):
        Stride of the convolution, as (stride_height, stride_width).
        Default is (1, 1).

    Returns:
    tuple:
        Gradients with respect to the input of the convolutional layer (`dA`),
        the weights (`dW`), and the biases (`db`):
        - dA (numpy.ndarray): Gradient of the cost with respect to the input
        (A_prev), of shape (m, h_prev, w_prev, c_prev).
        - dW (numpy.ndarray): Gradient of the cost with respect to the weights,
          of shape (kh, kw, c_prev, c_new).
        - db (numpy.ndarray): Gradient of the cost with respect to the biases,
          of shape (1, 1, 1, c_new).
    """
    # Retrieving dimensions from dZ
    m, h_new, w_new, c_new = dZ.shape

    # Retrieving dimensions from A_prev shape
    _, h_prev, w_prev, c_prev = A_prev.shape

    # Retrieving dimensions from W's shape
    kh, kw, _, _ = W.shape

    # Retrieving stride
    (sh, sw) = stride

    # Setting padding for valid
    pw, ph = 0, 0

    # bias calculation
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Setting padding for same
    if padding == 'same':
        ph = int(np.ceil(((h_prev-1)*sh+kh-h_prev)/2))
        pw = int(np.ceil(((w_prev-1)*sw+kw-w_prev)/2))

    # pad images
    A_prev = np.pad(A_prev,
                    pad_width=((0, 0),
                               (ph, ph),
                               (pw, pw),
                               (0, 0)),
                    mode='constant', constant_values=0)

    # Initializing dX, dW with the correct shapes
    dA = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)

    # Looping over vertical(h) and horizontal(w) axis of the output
    for z in range(m):
        for y in range(h_new):
            for x in range(w_new):
                # over every channel
                for v in range(c_new):
                    aux_W = W[:, :, :, v]
                    aux_dz = dZ[z, y, x, v]
                    dA[z, y*sh: y*sh+kh, x*sw: x*sw+kw, :] += aux_dz * aux_W
                    aux_A_prev = A_prev[z, y*sh: y*sh+kh, x*sw: x*sw+kw, :]
                    dW[:, :, :, v] += aux_A_prev * aux_dz

    # subtracting padding
    dA = dA[:, ph:dA.shape[1]-ph, pw:dA.shape[2]-pw, :]

    return dA, dW, db
