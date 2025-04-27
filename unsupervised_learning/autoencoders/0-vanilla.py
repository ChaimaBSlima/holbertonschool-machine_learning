#!/usr/bin/env python3
""" Task 0: 0. "Vanilla" Autoencoder  """
import keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder model with a specific architecture.

    An autoencoder is composed of two parts:
    - An encoder that maps the input to a compressed latent space
  representation.
    - A decoder that reconstructs the input from the latent space.

    The encoder and decoder are built using fully connected (Dense)
    layers.

    Args:
        input_dims (int): The dimensions of the input features.
        hidden_layers (list of int): List containing the number of
                                     nodes for each hidden layer in
                                     the encoder. The decoder will
                                     have the same layers in reverse
                                     order.
        latent_dims (int): The number of dimensions for the latent
                            space representation.

    Returns:
        encoder (keras.Model): The encoder part of the autoencoder.
        decoder (keras.Model): The decoder part of the autoencoder.
        autoencoder (keras.Model): The full autoencoder model combining
                                    encoder and decoder.

    Model Details:
        - Encoder:
            * Input layer with shape (input_dims,)
            * Several hidden Dense layers with 'relu' activation based
              on hidden_layers
            * Final Dense layer with 'relu' activation mapping to
              latent_dims
        - Decoder:
            * Input layer with shape (latent_dims,)
            * Hidden Dense layers mirroring the encoder's hidden
              layers (reversed)
            * Final Dense layer with 'sigmoid' activation mapping
              back to input_dims
        - Autoencoder:
            * Connects the encoder and decoder
            * Compiled with Adam optimizer and binary crossentropy
            loss
    """
    hidden_layers_length = len(hidden_layers)

    # encoder
    encoded = x = keras.Input(shape=(input_dims, ))
    for i in range(hidden_layers_length):
        x = keras.layers.Dense(hidden_layers[i], activation="relu")(x)
    h = keras.layers.Dense(latent_dims, activation="relu")(x)
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
