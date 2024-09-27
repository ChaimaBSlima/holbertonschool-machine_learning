#!/usr/bin/env python3
""" Task 5: 5. Multiple Kernels """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Perform a convolution operation on a batch of images using
    multiple kernels.

    This function applies a set of convolution kernels to a batch of images,
    allowing for configurable padding and stride, and outputs the convolved
    feature maps.

    Parameters:
    images (numpy.ndarray):
    A numpy array of shape (m, h, w, c) containing m images,
    each of height h, width w, and c channels.
    kernels (numpy.ndarray):
    A numpy array of shape (kh, kw, c, nc) representing the convolution
    kernels, where kh is the kernel height, kw is the kernel width, c is
    the number of channels, and nc is the number of kernels.
    padding (str or tuple):
    Either 'same' to maintain the original dimensions of the images,
    or a tuple of (ph, pw) representing height and width padding.
    stride (tuple):
    A tuple of (sh, sw) representing the vertical and horizontal strides.

    Returns:
    numpy.ndarray:
    A numpy array of shape (m, new_h, new_w, nc) containing the result of the
    convolution, where new_h and new_w are the height and width of the output
    feature maps, respectively.
    """
    c, w, = images.shape[3], images.shape[2]
    h, m = images.shape[1], images.shape[0]
    nc, kw, kh = kernels.shape[3], kernels.shape[1], kernels.shape[0]
    sw, sh = stride[1], stride[0]

    pw, ph = 0, 0

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    if isinstance(padding, tuple):
        # Extract required padding
        ph = padding[0]
        pw = padding[1]

    # pad images
    images = np.pad(images,
                    pad_width=((0, 0),
                               (ph, ph),
                               (pw, pw),
                               (0, 0)),
                    mode='constant', constant_values=0)

    new_h = int(((h + 2 * ph - kh) / sh) + 1)
    new_w = int(((w + 2 * pw - kw) / sw) + 1)

    # initialize convolution output tensor
    output = np.zeros((m, new_h, new_w, nc))

    # Loop over every pixel of the output
    for y in range(new_h):
        for x in range(new_w):
            # over every kernel
            for v in range(nc):
                # element-wise multiplication of the kernel and the image
                output[:, y, x, v] = \
                    (kernels[:, :, :, v] *
                     images[:,
                     y * sh: y * sh + kh,
                     x * sw: x * sw + kw,
                     :]).sum(axis=(1, 2, 3))

    return output
