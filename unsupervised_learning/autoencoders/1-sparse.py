#!/usr/bin/env python3
""" Task 1: 1. Sparse Autoencoder """
import keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder model with L1 activity regularization on
    the latent space.

    An autoencoder is composed of two parts:
    - An encoder that compresses the input into a lower-dimensional
    latent space.
    - A decoder that reconstructs the original input from the latent space.

    L1 regularization is applied on the latent space to encourage sparsity
    (many activations close to zero).

    Args:
        input_dims (int): The number of input features.
        hidden_layers (list of int): List containing the number of neurons
                                     for each hidden layer in the encoder.
                                     The decoder will mirror the encoder's
                                     structure in reverse.
        latent_dims (int): The dimensionality of the latent space
                          representation.
        lambtha (float): The L1 regularization parameter to apply to the latent
                        space layer.

    Returns:
        encoder (keras.Model): The encoder part of the autoencoder.
        decoder (keras.Model): The decoder part of the autoencoder.
        autoencoder (keras.Model): The complete autoencoder model combining
                                  encoder and decoder.

    Model Architecture:
        - Encoder:
            * Input layer with shape (input_dims,)
            * Hidden Dense layers with 'relu' activation according to
              hidden_layers
            * Latent Dense layer with 'relu' activation and L1 activity
              regularization
        - Decoder:
            * Input layer with shape (latent_dims,)
            * Hidden Dense layers mirroring the encoder's hidden layers
              (reversed)
            * Final Dense layer with 'sigmoid' activation to reconstruct
              the input
        - Autoencoder:
            * Sequential combination of encoder and decoder
            * Compiled with Adam optimizer and binary crossentropy loss
    """
    hidden_layers_length = len(hidden_layers)

    # encoder
    encoded = x = keras.Input(shape=(input_dims, ))
    regulizer = keras.regularizers.l1(lambtha)

    for i in range(hidden_layers_length):
        x = keras.layers.Dense(hidden_layers[i], activation="relu",)(x)
    h = keras.layers.Dense(latent_dims,
                           activation="relu",
                           activity_regularizer=regulizer)(x)
    encoder = keras.models.Model(inputs=encoded, outputs=h)

    # decoder
    decoded = y = keras.Input(shape=(latent_dims, ))
    for j in range((hidden_layers_length - 1), -1, -1):
        y = keras.layers.Dense(hidden_layers[j], activation="relu")(y)
    r = keras.layers.Dense(input_dims, activation="sigmoid")(y)
    decoder = keras.models.Model(inputs=decoded, outputs=r)

    # autoencoder
    inputs = keras.Input(shape=(input_dims, ))
    outputs = decoder(encoder(inputs))

    autoencoder = keras.models.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer="Adam", loss="binary_crossentropy")

    return encoder, decoder, autoencoder
