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
    initializer = K.initializers.HeNormal(seed=0)

    model = K.Sequential([
        X,
        K.layers.Conv2D(filters=6,
                        kernel_size=5,
                        padding='same',
                        activation='relu',
                        kernel_initializer=initializer),
        K.layers.MaxPool2D(pool_size=2, strides=2),
        K.layers.Conv2D(filters=16,
                        kernel_size=5,
                        activation='relu',
                        kernel_initializer=initializer),
        K.layers.MaxPool2D(pool_size=2, strides=2),
        K.layers.Flatten(),
        K.layers.Dense(units=120,
                       activation='relu',
                       kernel_initializer=initializer),
        K.layers.Dense(units=84,
                       activation='relu',
                       kernel_initializer=initializer),
        K.layers.Dense(units=10,
                       activation='softmax',
                       kernel_initializer=initializer)
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])

    return model
