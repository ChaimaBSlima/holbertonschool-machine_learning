#!/usr/bin/env python3
""" Task 4: 4. LeNet-5 (Tensorflow 1) """
import tensorflow as tf


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
    # implement He et. al initialization for the layer weights
    initializer = \
        tf.contrib.layers.variance_scaling_initializer()

    # Conv layers
    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    layer0 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                              padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              name='layer')(x)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    layer1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2))(layer0)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    layer2 = tf.layers.Conv2D(filters=16,
                              kernel_size=(5, 5),
                              padding='valid',
                              activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              name='layer')(layer1)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    layer3 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2))(layer2)

    # Flattening between conv and dense layers
    layer3 = tf.layers.Flatten()(layer3)

    # Fully connected (Dense) layers
    # Fully connected layer with 120 nodes
    layer4 = tf.layers.Dense(units=120,
                             activation=tf.nn.relu,
                             kernel_initializer=initializer,
                             name='layer')(layer3)

    # Fully connected layer with 84 nodes
    layer5 = tf.layers.Dense(units=84,
                             activation=tf.nn.relu,
                             kernel_initializer=initializer,
                             name='layer')(layer4)

    # Fully connected softmax output layer with 10 nodes
    layer6 = tf.layers.Dense(units=10,
                             kernel_initializer=initializer,
                             name='layer')(layer5)

    # loss
    loss = tf.losses.softmax_cross_entropy(y, layer6)

    # prediction
    y_pred = tf.nn.softmax(layer6)

    # train_op
    train_op = tf.train.AdamOptimizer(name='Adam').minimize(loss)

    # accuracy
    # from one y_pred one_hot to tag
    y_pred_t = tf.argmax(y_pred, 1)
    # from y one_hot to tag
    y_t = tf.argmax(y, 1)
    # comparison vector between tags (TRUE/FALSE)
    equal = tf.equal(y_pred_t, y_t)
    # average hits
    acc = tf.reduce_mean(tf.cast(equal, tf.float32))

    return y_pred, train_op, loss, acc
