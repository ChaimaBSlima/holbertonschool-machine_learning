#!/usr/bin/env python3
"""Performs PCA color augmentation
(AlexNet style) using TensorFlow."""
import tensorflow as tf


def pca_color(image, alphas):
    """
    Performs PCA color augmentation (AlexNet style).

    Args:
        image: A 3D tf.Tensor (H, W, 3) containing the image.
        alphas: Tuple of length 3 with the perturbation factors.

    Returns:
        The PCA color-augmented image as a tf.Tensor.
    """
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)

    # Flatten pixels to (num_pixels, 3)
    flat_img = tf.reshape(image, [-1, 3])

    # Compute covariance matrix (3x3)
    mean = tf.reduce_mean(flat_img, axis=0, keepdims=True)
    centered = flat_img - mean
    cov = tf.matmul(centered, centered,
                    transpose_a=True) / tf.cast(
                        tf.shape(centered)[0] - 1, tf.float32)

    # Eigen decomposition
    eigvals, eigvecs = tf.linalg.eigh(cov)

    # Sort eigenvalues & eigenvectors descending
    idx = tf.argsort(eigvals, direction='DESCENDING')
    eigvals = tf.gather(eigvals, idx)
    eigvecs = tf.gather(eigvecs, idx, axis=1)

    # Compute perturbation: eigvecs @ (eigvals * alphas)
    delta = tf.matmul(
        eigvecs,
        tf.reshape(eigvals * tf.constant(alphas, dtype=tf.float32), [-1, 1])
    )
    delta = tf.reshape(delta, [1, 3])  # shape (1,3)

    # Add delta to all pixels
    flat_img = flat_img + delta

    # Reshape back to image
    img_out = tf.reshape(flat_img, shape)

    # Clip to [0,255] and convert back to uint8
    img_out = tf.clip_by_value(img_out, 0.0, 255.0)
    return tf.cast(img_out, tf.uint8)
