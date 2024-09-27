#!/usr/bin/env python3
""" Task 6: 6. Pooling """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Perform pooling operation on a batch of images.

    This function applies a pooling layer to a batch of images using
    either max or average pooling.

    Parameters:
    images (numpy.ndarray):
    A numpy array of shape (m, h, w, c) containing m images,
    each of height h, width w, and c channels.
    kernel_shape (tuple):
    A tuple representing the height and width of the pooling kernel (kh, kw).
    stride (tuple):
    A tuple of (sh, sw) representing the vertical and horizontal strides.
    mode (str):
    A string indicating the pooling mode. Either 'max' for max pooling or
    'avg' for average pooling.

    Returns:
    numpy.ndarray:
    A numpy array of shape (m, new_h, new_w, c) containing the pooled images,
    where new_h and new_w are the height and width of the output images,
    respectively.
    """
    c, w, = images.shape[3], images.shape[2]
    h, m = images.shape[1], images.shape[0]
    kw, kh = kernel_shape[1], kernel_shape[0]
    sw, sh = stride[1], stride[0]

    new_h = int(((h - kh) / sh) + 1)
    new_w = int(((w - kw) / sw) + 1)

    # initialize convolution output tensor
    output = np.zeros((m, new_h, new_w, c))

    # Loop over every pixel of the output
    for x in range(new_w):
        for y in range(new_h):
            # element-wise multiplication of the kernel and the image
            if mode == 'max':
                output[:, y, x, :] = \
                    np.max(images[:,
                                  y * sh: y * sh + kh,
                                  x * sw: x * sw + kw], axis=(1, 2))
            if mode == 'avg':
                output[:, y, x, :] = \
                    np.mean(images[:,
                                   y * sh: y * sh + kh,
                                   x * sw: x * sw + kw], axis=(1, 2))

    return output
