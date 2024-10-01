#!/usr/bin/env python3
""" Task 4: 4. LeNet-5 (Tensorflow 1) """
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds the LeNet-5 architecture using TensorFlow for digit classification.

    Args:
    x (tf.Tensor):
        Input tensor of shape (m, 32, 32, c), where:
        - m is the number of examples,
        - 32x32 is the spatial size of the images,
        - c is the number of channels
          (e.g., 1 for grayscale images, 3 for RGB).

    y (tf.Tensor):
        One-hot encoded labels tensor of shape (m, 10), where:
        - m is the number of examples,
        - 10 is the number of classes for classification.

    Returns:
    y_pred (tf.Tensor):
        Tensor containing the softmax predictions for each input,
        of shape (m, 10).

    train_op (tf.Operation):
        TensorFlow operation for training, using the Adam optimizer
        to minimize the loss.

    loss (tf.Tensor):
        Tensor representing the softmax cross-entropy loss
        predictions and labels.

    acc (tf.Tensor):
        Tensor representing the accuracy of the model,
        as the percentage of correct predictions.

    Model Architecture:
    - Input: A tensor representing the input images
    (e.g., 32x32 pixels for grayscale).
    - Layer 1: Conv2D with 6 filters of size 5x5, using 'same' padding,
              followed by ReLU activation.
    - Layer 2: MaxPooling2D with a 2x2 pool size and 2x2 stride.
    - Layer 3: Conv2D with 16 filters of size 5x5, using 'valid'
              padding, followed by ReLU activation.
    - Layer 4: MaxPooling2D with a 2x2 pool size and 2x2 stride.
    - Layer 5: Fully connected (Dense) layer with 120 units and
               ReLU activation.
    - Layer 6: Fully connected (Dense) layer with 84 units and ReLU activation.
    - Output Layer: Fully connected (Dense) layer with 10 units
                    (number of classes) and softmax activation.

    Loss:
    - The model uses softmax cross-entropy loss for classification.

    Optimization:
    - The model is trained using the Adam optimizer.

    Accuracy:
    - The accuracy is computed by comparing the predicted class with
    the actual labels.
    """
    weights_init = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv_1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding='same',
        activation="relu",
        kernel_initializer=weights_init
    )(x)

    pool_1 = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    )(conv_1)

    conv_2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding='valid',
        activation="relu",
        kernel_initializer=weights_init
    )(pool_1)

    pool_2 = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    )(conv_2)

    flat = tf.layers.Flatten()(pool_2)

    layer_1 = tf.layers.Dense(
        units=120,
        activation="relu",
        name="layer",
        kernel_initializer=weights_init
    )(flat)

    layer_2 = tf.layers.Dense(
        units=84,
        activation="relu",
        name="layer",
        kernel_initializer=weights_init
    )(layer_1)

    output = tf.layers.Dense(
        units=10,
        activation=None,
        name="layer",
        kernel_initializer=weights_init
    )(layer_2)

    losses = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=output
    )

    comparation = tf.math.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(comparation, tf.float32))

    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(losses)

    out = tf.nn.softmax(output)

    return out, train, losses, accuracy
