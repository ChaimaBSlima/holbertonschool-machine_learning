#!/usr/bin/env python3
""" Task 5: 5. LeNet-5 (Keras)  """
from tensorflow import keras as K


def lenet5(X):
    """
    Creates a LeNet-5 model using Keras.

    LeNet-5 is a convolutional neural network (CNN) architecture designed for
    handwritten digit recognition. This implementation follows the original
    architecture but can be adapted for various classification tasks.

    Args:
        X (tf.Tensor): A Keras tensor that serves as the input for the model.
                       It should have a shape compatible with the model's
                       requirements.
                       (e.g., (batch_size, height, width, channels)).

    Returns:
        tf.keras.Model: A Keras model instance representing the LeNet-5
        architecture. The model is compiled with Adam optimizer and
        categorical cross-entropy loss function, suitable for multi-class
         classification problems.
    """
    initializer = K.initializers.HeNormal()

    conv_1 = K.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding="same",
        kernel_initializer=initializer,
        activation="relu"
    )(X)

    pool_1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv_1)

    conv_2 = K.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding="valid",
        kernel_initializer=initializer,
        activation="relu"
    )(pool_1)

    pool_2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv_2)

    flat = K.layers.Flatten()(pool_2)

    layer_1 = K.layers.Dense(120, activation='relu',
                             kernel_initializer=K.initializers.HeNormal(seed=None))(flat)
    layer_2 = K.layers.Dense(84, activation='relu',
                             kernel_initializer=K.initializers.HeNormal(seed=None))(layer_1)
    layer_3 = K.layers.Dense(10, activation='softmax',
                             kernel_initializer=K.initializers.HeNormal(seed=None))(layer_2)

    model = K.Model(inputs=X, outputs=layer_3)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model
