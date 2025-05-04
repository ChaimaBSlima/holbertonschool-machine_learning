#!/usr/bin/env python3
""" Task 3: 3. Generating faces """
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """
    Build and return a convolutional GAN architecture:

    Generator:
      - Input: latent vector of shape (16,)
      - Dense to 2048 units, tanh activation
      - Reshape to (2, 2, 512)
      - Three upsampling blocks:
          * UpSampling2D
          * Conv2D (tanh activations, BatchNormalization)
      - Final output: 16×16×1 image with tanh activation

    Discriminator (Critic):
      - Input: 16×16×1 image
      - Four convolutional blocks:
          * Conv2D (padding='same')
          * MaxPooling2D
          * tanh activation
      - Flatten and Dense to single scalar output

    Returns
    -------
    generator : keras.Model
        The generator model mapping latent vectors to images.
    discriminator : keras.Model
        The discriminator model scoring input images.
    """
    # Generator definition
    def get_generator():
        inputs = keras.Input(shape=(16,), name='gen_input')
        x =\
            keras.layers.Dense(2048, activation='tanh',
                               name='gen_dense')(inputs)
        x = keras.layers.Reshape((2, 2, 512), name='gen_reshape')(x)
        # Upsample to 4×4
        x = keras.layers.UpSampling2D(name='gen_upsample1')(x)
        x = keras.layers.Conv2D(64, 3, padding='same', name='gen_conv1')(x)
        x = keras.layers.BatchNormalization(name='gen_bn1')(x)
        x = keras.layers.Activation('tanh', name='gen_act1')(x)
        # Upsample to 8×8
        x = keras.layers.UpSampling2D(name='gen_upsample2')(x)
        x = keras.layers.Conv2D(16, 3, padding='same', name='gen_conv2')(x)
        x = keras.layers.BatchNormalization(name='gen_bn2')(x)
        x = keras.layers.Activation('tanh', name='gen_act2')(x)
        # Upsample to 16×16
        x = keras.layers.UpSampling2D(name='gen_upsample3')(x)
        x = keras.layers.Conv2D(1, 3, padding='same', name='gen_conv3')(x)
        x = keras.layers.BatchNormalization(name='gen_bn3')(x)
        outputs = keras.layers.Activation('tanh', name='gen_output')(x)
        return keras.Model(inputs, outputs, name='generator')

    # Discriminator definition
    def get_discriminator():
        inputs = keras.Input(shape=(16, 16, 1), name='disc_input')
        # Block 1: 16→8
        x = keras.layers.Conv2D(32, 3, padding='same',
                                name='disc_conv1')(inputs)
        x = keras.layers.MaxPooling2D(name='disc_pool1')(x)
        x = keras.layers.Activation('tanh', name='disc_act1')(x)
        # Block 2: 8→4
        x = keras.layers.Conv2D(64, 3, padding='same', name='disc_conv2')(x)
        x = keras.layers.MaxPooling2D(name='disc_pool2')(x)
        x = keras.layers.Activation('tanh', name='disc_act2')(x)
        # Block 3: 4→2
        x = keras.layers.Conv2D(128, 3, padding='same', name='disc_conv3')(x)
        x = keras.layers.MaxPooling2D(name='disc_pool3')(x)
        x = keras.layers.Activation('tanh', name='disc_act3')(x)
        # Block 4: 2→1
        x = keras.layers.Conv2D(256, 3, padding='same', name='disc_conv4')(x)
        x = keras.layers.MaxPooling2D(name='disc_pool4')(x)
        x = keras.layers.Activation('tanh', name='disc_act4')(x)
        x = keras.layers.Flatten(name='disc_flatten')(x)
        outputs = keras.layers.Dense(1, name='disc_output')(x)
        return keras.Model(inputs, outputs, name='discriminator')

    return get_generator(), get_discriminator()
