#!/usr/bin/env python3
""" Task 2: 2. Convolutional Autoencoder  """
import keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder model for image-like data.

    A convolutional autoencoder consists of:
    - An encoder that compresses the input image into a low-dimensional feature
    representation using convolution and pooling layers.
    - A decoder that reconstructs the original image from the compressed
    feature representation using convolution and upsampling layers.

    Args:
        input_dims (tuple): Dimensions of the input data
                            (height, width, channels).
        filters (list of int): List of the number of filters for each
                            convolutional layer in the encoder. The decoder
                            mirrors this structure in reverse.
        latent_dims (tuple): Dimensions of the latent space
                            (should match the encoder output shape).

    Returns:
        encoder (keras.Model): The encoder part of the autoencoder.
        decoder (keras.Model): The decoder part of the autoencoder.
        auto (keras.Model): The full autoencoder model combining encoder and
                            decoder.

    Model Architecture:
        - Encoder:
            * Input layer with shape input_dims
            * Several Conv2D layers with 'relu' activation and (3x3) kernels
            * MaxPooling2D layers after each convolution to reduce spatial
                dimensions
        - Latent space:
            * Final output of the encoder without explicit compression to
            a dense layer
        - Decoder:
            * Input layer with shape latent_dims
            * Several Conv2D layers with 'relu' activation and (3x3) kernels
            * UpSampling2D layers after each convolution to increase spatial
                dimensions
            * Final Conv2D layer with 'sigmoid' activation to reconstruct the
                input image
        - Autoencoder:
            * Chains encoder and decoder together
            * Compiled with Adam optimizer and binary crossentropy loss

    Notes:
        - The model uses 'same' padding to preserve spatial dimensions during
        convolution and pooling
        (except for one 'valid' convolution in the decoder).
        - The decoder architecture mirrors the encoder in reverse order.
        - Suitable for grayscale or RGB images.
    """
    input_encoder = keras.Input(shape=input_dims)
    input_decoder = keras.Input(shape=latent_dims)

    # Encoder model
    encoded = keras.layers.Conv2D(filters[0],
                                  kernel_size=(3, 3),
                                  padding='same',
                                  activation='relu')(input_encoder)
    encoded = keras.layers.MaxPool2D((2, 2),
                                     padding='same')(encoded)
    for enc in range(1, len(filters)):
        encoded = keras.layers.Conv2D(filters[enc],
                                      kernel_size=(3, 3),
                                      padding='same',
                                      activation='relu')(encoded)
        encoded = keras.layers.MaxPool2D((2, 2),
                                         padding='same')(encoded)

    # Latent layer
    latent = encoded
    encoder = keras.Model(inputs=input_encoder, outputs=latent)

    # Decoded model
    decoded = keras.layers.Conv2D(filters[-1],
                                  kernel_size=(3, 3),
                                  padding='same',
                                  activation='relu')(input_decoder)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    for dec in range(len(filters) - 2, 0, -1):
        decoded = keras.layers.Conv2D(filters[dec],
                                      kernel_size=(3, 3),
                                      padding='same',
                                      activation='relu')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    last = keras.layers.Conv2D(filters[0],
                               kernel_size=(3, 3),
                               padding='valid',
                               activation='relu')(decoded)
    last = keras.layers.UpSampling2D((2, 2))(last)
    last = keras.layers.Conv2D(input_dims[-1],
                               kernel_size=(3, 3),
                               padding='same',
                               activation='sigmoid')(last)
    decoder = keras.Model(inputs=input_decoder, outputs=last)

    encoder_output = encoder(input_encoder)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=input_encoder, outputs=decoder_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
