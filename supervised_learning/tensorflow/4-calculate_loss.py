#!/usr/bin/env python3
""" Task 4: 4. Loss """
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss between
    true labels and predictions.

    Parameters:
    y : tf.Tensor
        A tensor of true labels, with shape [batch_size, num_classes].
        The labels should be in one-hot encoded format.
    y_pred : tf.Tensor
        A tensor of predicted logits from the network, with shape
        [batch_size, num_classes].
        These are the raw, unnormalized scores output by the network.

    Returns:
    tf.Tensor
        A scalar tensor representing the softmax cross-entropy loss
        between the true labels and the predicted logits.
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
