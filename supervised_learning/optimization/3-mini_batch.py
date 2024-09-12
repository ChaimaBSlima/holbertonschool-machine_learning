#!/usr/bin/env python3
""" Task 3: 3. Mini-Batch """
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a preloaded neural network model using mini-batch gradient descent.

    Args:
        X_train (numpy.ndarray):
            Training data of shape (m, 784), where:
            - m is the number of training data points.
            - 784 is the number of input features.
        Y_train (numpy.ndarray):
            One-hot training labels of shape (m, 10), where:
            - m is the number of training data points.
            - 10 is the number of classes.
        X_valid (numpy.ndarray):
            Validation data of shape (m, 784), where:
            - m is the number of validation data points.
            - 784 is the number of input features.
        Y_valid (numpy.ndarray):
            One-hot validation labels of shape (m, 10), where:
            - m is the number of validation data points.
            - 10 is the number of classes.
        batch_size (int):
            Number of data points in each mini-batch for training.
        epochs (int):
            Number of full passes over the entire dataset during training.
        load_path (str):
            Filepath from which to load the pre-trained model checkpoint.
        save_path (str):
            Filepath where the trained model should be saved after training.

    Returns:
        str: The file path where the trained model is saved.
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        steps = X_train.shape[0] // batch_size
        if steps % batch_size != 0:
            steps = steps + 1
            extra = True
        else:
            extra = False

        for epoch in range(epochs + 1):
            train_cost, train_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                for step_number in range(steps):
                    start = step_number * batch_size

                    if step_number == steps - 1 and extra:
                        end = X_train.shape[0]
                    else:
                        end = step_number * batch_size + batch_size

                    X = X_shuffled[start:end]
                    Y = Y_shuffled[start:end]
                    sess.run(train_op, feed_dict={x: X, y: Y})

                    if step_number != 0 and (step_number + 1) % 100 == 0:
                        print("\tStep {}:".format(step_number + 1))
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X, y: Y})
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        return saver.save(sess, save_path)
