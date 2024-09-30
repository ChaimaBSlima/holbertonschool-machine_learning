#!/usr/bin/env python3
""" Task 0: 0. Convolutional Forward Prop """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation for a convolutional layer.

    Args:
    A_prev (numpy.ndarray):
        Input data of shape (m, h_prev, w_prev, c_prev) where:
        - m is the number of examples,
        - h_prev is the height of the previous layer,
        - w_prev is the width of the previous layer,
        - c_prev is the number of channels in the previous layer.

    W (numpy.ndarray):
        Weights (kernels) for the convolutional layer,
        of shape (kh, kw, c_prev, c_new) where:
        - kh is the kernel height,
        - kw is the kernel width,
        - c_prev is the number of channels in the previous layer,
        - c_new is the number of output channels (filters).

    b (numpy.ndarray):
        Biases for the convolutional layer,
        of shape (1, 1, 1, c_new).

    activation (function):
        The activation function to apply to the convolution output
        (e.g., `np.relu` for ReLU).

    padding (str, optional):
        Padding type, either 'same' or 'valid'.
        'same' adds padding to the input to ensure the output
        has the same dimensions as the input.
        Default is 'same'.

    stride (tuple, optional):
        Stride of the convolution, as (stride_height, stride_width).
        Default is (1, 1).

    Returns:
    numpy.ndarray:
        Output of the convolutional layer after applying the
        specified activation function.
        The shape of the output is (m, n_h, n_w, c_new) where:
        - n_h is the height of the output,
        - n_w is the width of the output,
        - c_new is the number of output channels (filters).
    """

    # Retrieving dimensions from A_prev shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    # Retrieving dimensions from W's shape
    (kh, kw, c_prev, c_new) = W.shape

    # Retrieving stride
    (sh, sw) = stride

    pw, ph = 0, 0

    # Compute the output dimensions
    n_h = int(((h_prev + 2 * ph - kh) / sh) + 1)
    n_w = int(((w_prev + 2 * pw - kw) / sw) + 1)

    # Setting padding for same
    if padding == 'same':
        if kh % 2 == 0:
            ph = int((h_prev * sh + kh - h_prev) / 2)
            n_h = int(((h_prev + 2 * ph - kh) / sh))
        else:
            ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
            n_h = int(((h_prev + 2 * ph - kh) / sh) + 1)

        if kw % 2 == 0:
            pw = int((w_prev * sw + kw - w_prev) / 2)
            n_w = int(((w_prev + 2 * pw - kw) / sw))
        else:
            pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
            n_w = int(((w_prev + 2 * pw - kw) / sw) + 1)

    images = np.pad(A_prev,
                    pad_width=((0, 0),
                               (ph, ph),
                               (pw, pw),
                               (0, 0)),
                    mode='constant', constant_values=0)

    # Initialize the output with zeros
    output = np.zeros((m, n_h, n_w, c_new))

    # Looping over vertical and horizontal axis of output volume
    for y in range(n_h):
        for x in range(n_w):
            # over every channel
            for v in range(c_new):
                # element-wise multiplication of the kernel and the image
                output[:, y, x, v] = \
                    (W[:, :, :, v] *
                     images[:,
                            y * sh: y * sh + kh,
                            x * sw: x * sw + kw,
                            :]).sum(axis=(1, 2, 3))

                output[:, y, x, v] = \
                    (activation(output[:, y, x, v] +
                                b[0, 0, 0, v]))

    return output
