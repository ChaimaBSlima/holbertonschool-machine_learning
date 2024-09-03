#!/usr/bin/env python3
""" Task 3: 3. Accuracy """
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of predictions compared to the true labels.

    Parameters:
    y : tf.Tensor
        A tensor of true labels, with shape [batch_size, num_classes].
    y_pred : tf.Tensor
        A tensor of predicted labels from the network,
        with shape [batch_size, num_classes].

    Returns:
    tf.Tensor
        A scalar tensor representing the accuracy of the predictions,
        where accuracy is the mean of the correct predictions.
    """
    correct_prediction = tf.equal(tf.argmax(y, 1),
                                  tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
