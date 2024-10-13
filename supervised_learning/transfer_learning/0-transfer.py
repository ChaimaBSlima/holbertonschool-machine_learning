#!/usr/bin/env python3
""" Transfer Learning """

import tensorflow.keras as K
import matplotlib.pyplot as plt


def preprocess_data(X, Y):
    """
    Preprocesses the input data for compatibility with the ResNet50 model.

    Parameters:
    X : numpy.ndarray
        Input image data, expected to be in the form of a NumPy array.
    Y : numpy.ndarray
        Labels corresponding to the input data, also in the
        form of a NumPy array.

    Returns:
    x_test : numpy.ndarray
        Preprocessed image data ready for use in ResNet50,
        with appropriate scaling and formatting.
    y_test : numpy.ndarray
        One-hot encoded labels corresponding to the input data,
        suitable for classification tasks.
    """
    x_test = K.applications.resnet50.preprocess_input(X)
    y_test = K.utils.to_categorical(Y)
    return x_test, y_test


# script should not run when file is imported
if __name__ == '__main__':

    def resize_image(X):
        """
      Resizes input images to 224x224 to be compatible with the ResNet50 model.

        Parameters:
        X : numpy.ndarray
            Input image data to be resized.

        Returns:
        resized_X : numpy.ndarray
            Resized image data to match the input shape required by ResNet50.
        """
        return K.backend.resize_images(
            X, 7, 7, data_format='channels_last',
            interpolation='bilinear'
        )

    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
    optimizer = K.optimizers.Adam(learning_rate=0.00001)
    model = K.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # freeze layers
    model.trainable = False
    input = K.Input(shape=(32, 32, 3))
    lambda_ = K.layers.Lambda(resize_image)(input)
    x = model(lambda_, training=False)
    x = K.layers.Flatten()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(128, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(64, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(10, activation='softmax')(x)
    model = K.Model(input, x)
    model.summary()
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    epochs = 20
    batch_size = 256
    xt_prep, yt_prep = preprocess_data(X_train, Y_train)
    xv_prep, yv_prep = preprocess_data(X_valid, Y_valid)
    my_callbacks = [K.callbacks.ModelCheckpoint(
                    filepath='cifar10.keras', save_best_only=True)]

    history = model.fit(xt_prep, yt_prep,
                        batch_size=batch_size,
                        validation_data=(xv_prep, yv_prep),
                        epochs=epochs,
                        callbacks=my_callbacks)
