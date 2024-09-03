#!/usr/bin/env python3
""" Task 6: 6. Train """
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Parameters:
    X_train : numpy.ndarray
        Training input data with shape [num_samples, num_features].
    Y_train : numpy.ndarray
        Training labels with shape [num_samples, num_classes].
        Labels should be one-hot encoded.
    X_valid : numpy.ndarray
        Validation input data with shape [num_samples, num_features].
    Y_valid : numpy.ndarray
        Validation labels with shape [num_samples, num_classes].
        Labels should be one-hot encoded.
    layer_sizes : list of int
        A list where each element represents the number of neurons in a layer.
    activations : list of callable
        A list where each element is an activation function to be
        applied to each layer.
    alpha : float
        The learning rate for the gradient descent optimizer.
    iterations : int
        The number of iterations (epochs) to train the network.
    save_path : str, optional
        Path to save the trained model checkpoint.
        Default is "/tmp/model.ckpt".

    Returns:
    str
        The path where the trained model is saved.
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    for step in range(iterations + 1):
        t_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        t_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
        v_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        v_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        if step % 100 == 0 or step == iterations:
            print("After {} iterations:".format(step))
            print("\tTraining Cost: {}".format(t_cost))
            print("\tTraining Accuracy: {}".format(t_acc))
            print("\tValidation Cost: {}".format(v_cost))
            print("\tValidation Accuracy: {}".format(v_acc))
        if step < iterations:
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
    return saver.save(sess, save_path)
