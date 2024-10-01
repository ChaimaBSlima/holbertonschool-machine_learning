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
    # Set the initializer with the specified seed
    he_normal_initializer = K.initializers.HeNormal(seed=0)

    # Define the model
    model = K.Sequential([
        K.layers.Conv2D(6, (5, 5), padding='same', kernel_initializer=he_normal_initializer, activation='relu', input_shape=(28, 28, 1)),
        K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        K.layers.Conv2D(16, (5, 5), padding='valid', kernel_initializer=he_normal_initializer, activation='relu'),
        K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        K.layers.Flatten(),  # Flatten the output for the fully connected layers
        K.layers.Dense(120, kernel_initializer=he_normal_initializer, activation='relu'),
        K.layers.Dense(84, kernel_initializer=he_normal_initializer, activation='relu'),
        K.layers.Dense(10, kernel_initializer=he_normal_initializer, activation='softmax')
    ])

    # Compile the model using Adam optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
