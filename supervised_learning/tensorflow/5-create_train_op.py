#!/usr/bin/env python3
""" Task 5: 5. Train_Op """
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_train_op(loss, alpha):
    """
    Creates a training operation for optimizing the
    network using gradient descent.

    Parameters:
    -----------
    loss : tf.Tensor
        A scalar tensor representing the loss value that the optimizer
        aims to minimize.
    alpha : float
        The learning rate for the gradient descent optimizer,
        which controls the step size during optimization.

    Returns:
    --------
    tf.Operation
        A TensorFlow operation that applies gradients
        to minimize the loss using Gradient Descent.
    """
    opt = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return opt
