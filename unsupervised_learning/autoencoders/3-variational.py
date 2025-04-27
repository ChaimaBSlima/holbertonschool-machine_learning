#!/usr/bin/env python3
""" Task 3: 3. Variational Autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder model with a specific architecture.
    A variational autoencoder is a type of autoencoder that learns a
    probabilistic representation of the input data in the latent space.
    It consists of an encoder that maps the input to a latent space
    representation and a decoder that reconstructs the input from the
    latent space. The encoder outputs two vectors: the mean and the
    log variance of the latent space distribution. The decoder samples
    from this distribution to generate the reconstructed output.
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
    Model Architecture:
        - Encoder:
            * Input layer with shape (input_dims,)
            * Hidden Dense layers with 'relu' activations
            * Two output Dense layers (z_mean and z_log_sigma) with no
                activation
            * Sampling layer using reparameterization trick:
                z = z_mean + exp(z_log_sigma / 2) * epsilon
        - Decoder:
            * Input layer with shape (latent_dims,)
            * Hidden Dense layers mirroring the encoder
            * Final Dense layer with 'sigmoid' activation to reconstruct input
        - VAE:
            * Connects encoder and decoder
            * Compiled with Adam optimizer and a custom VAE loss:
                * Binary cross-entropy loss for reconstruction
                * KL divergence loss for latent space regularization

    Notes:
        - The KL divergence term regularizes the latent space to resemble
            a standard normal distribution.
        - The reparameterization trick (via the Lambda layer) allows
        backpropagation through the sampling operation.
    """
    input_encoder = keras.Input(shape=(input_dims, ))
    input_decoder = keras.Input(shape=(latent_dims, ))

    # Encoder model
    encoded = keras.layers.Dense(hidden_layers[0],
                                 activation='relu')(input_encoder)
    for enc in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[enc],
                                     activation='relu')(encoded)

    # Latent layer
    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(encoded)

    def sample_z(args):
        """
        Samples a point from the latent space using the
        reparameterization trick.

        This function is used in Variational Autoencoders (VAEs) to
        allow backpropagation  through a stochastic sampling operation.

        Args:
            args (tuple): A tuple containing two tensors:
                - mu: Tensor representing the mean of the latent normal
                    distribution.
                - sigma: Tensor representing the log-variance of the latent
                    normal distribution.

        Returns:
            Tensor: A sampled latent vector z based on the input mean
                    and log-variance.

        Sampling formula:
            z = mu + exp(sigma / 2) * epsilon
        where epsilon is drawn from a standard normal distribution N(0, I).

        Purpose:
            - Introduce randomness into the encoding process.
            - Ensure the sampling operation is differentiable with respect
            to mu and sigma by applying the "reparameterization trick"
            (sampling from a standard normal and shifting/scaling
            by learned parameters).
        """
        mu, sigma = args
        batch = keras.backend.shape(mu)[0]
        dim = keras.backend.int_shape(mu)[1]
        eps = keras.backend.random_normal(shape=(batch, dim))
        return mu + keras.backend.exp(sigma / 2) * eps

    z = keras.layers.Lambda(sample_z,
                            output_shape=(latent_dims,))([z_mean, z_log_sigma])

    encoder = keras.Model(inputs=input_encoder,
                          outputs=[z, z_mean, z_log_sigma])

    # Decoded model
    decoded = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(input_decoder)
    for dec in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[dec],
                                     activation='relu')(decoded)
    last = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=input_decoder, outputs=last)

    encoder_output = encoder(input_encoder)[0]
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=input_encoder, outputs=decoder_output)

    def vae_loss(x, x_decoded_mean):
        """
        Computes the loss function for the Variational Autoencoder (VAE).

        The VAE loss is composed of two parts:
        1. Reconstruction loss (xent_loss):
            - Measures how well the decoded outputs match the original
                inputs.
            - Calculated using binary cross-entropy between the input and
                its reconstruction.
            - Summed across all features for each sample.
            - Encourages the model to learn a good representation of the
                input data.
            - Suitable for binary or normalized data (e.g., pixel values
                between 0 and 1).
            - The formula used:
                xent_loss = -sum(x * log(x_decoded_mean) + (1 - x) *
                log(1 - x_decoded_mean))
            - This is equivalent to the binary cross-entropy loss
                function in Keras.
            - The loss is averaged across the batch size.
            - The loss is summed across all features for each sample.
            - The loss is computed using the Keras backend function
                `binary_crossentropy`.

        2. Kullback-Leibler (KL) divergence loss (kl_loss):
            - Measures how much the learned latent distribution
            (parameterized by z_mean and z_log_sigma)
            deviates from the standard normal distribution N(0, I).
            - Encourages the latent space to be smooth and continuous.

        Args:
            x (Tensor): Original input tensor.
            x_decoded_mean (Tensor): Reconstructed input tensor
                                    (output of the decoder).

        Returns:
            Tensor: The combined VAE loss
            (reconstruction loss + KL divergence loss),
            averaged across the batch.

        Notes:
            - `z_mean` and `z_log_sigma` must be accessible from the
                outer scope.
            - The KL divergence formula used:
                KL = -0.5 * sum(1 + log_var - mean² - exp(log_var))
        """
        xent_loss = keras.backend.binary_crossentropy(x, x_decoded_mean)
        xent_loss = keras.backend.sum(xent_loss, axis=1)
        kl_loss = - 0.5 * keras.backend.mean(
            1 + z_log_sigma - keras.backend.square(z_mean) - keras.backend.exp(
                z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
