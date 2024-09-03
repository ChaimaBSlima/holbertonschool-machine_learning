#!/usr/bin/env python3
""" Task 7: 7. Evaluate """
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def evaluate(X, Y, save_path):
    """
    Evaluates the performance of a neural network using a saved model.

    Parameters:
    X : numpy.ndarray
        Input data for evaluation, with shape [num_samples, num_features].
    Y : numpy.ndarray
        One-hot encoded labels for the input data, with
        [num_samples, num_classes].
    save_path : str
        The path to the model checkpoint file from which
        to restore the model.

    Returns:
    tuple of (numpy.ndarray, float, float)
        - The predicted outputs for the input data.
        - The accuracy of the model on the input data.
        - The loss value of the model on the input data.
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(save_path))
        saver.restore(sess, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        eval_y_pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        eval_accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        eval_loss = sess.run(loss, feed_dict={x: X, y: Y})
        return eval_y_pred, eval_accuracy, eval_loss
