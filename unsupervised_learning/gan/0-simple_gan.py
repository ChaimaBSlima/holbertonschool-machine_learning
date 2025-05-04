#!/usr/bin/env python3
""" Task 0: 0. Simple GAN"""
import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """
    A simple Generative Adversarial Network (GAN) model that
    alternates training
    between a discriminator and a generator.

    Parameters
    ----------
    generator : keras.Model
        The neural network model generating fake samples from latent vectors.
    discriminator : keras.Model
        The neural network model distinguishing real vs. fake samples.
    latent_generator : callable
        Function mapping an integer batch size to a tensor of latent samples.
    real_examples : tf.Tensor
        A tensor containing real data samples for discriminator training.
    batch_size : int, optional (default=200)
        Number of samples per training batch.
    disc_iter : int, optional (default=2)
        Number of discriminator update steps per generator update.
    learning_rate : float, optional (default=0.005)
        Learning rate for both optimizers.
    """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.005):
        super().__init__()

        # Core components
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples

        # Hyperparameters
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # Generator loss: mean squared error against label +1
        self.generator.loss = lambda pred: \
            tf.keras.losses.MeanSquaredError()(pred, tf.ones_like(pred))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        # Compile generator (needed for Keras internals like history tracking)
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        # Discriminator loss: MSE(real_pred, +1) + MSE(fake_pred, -1)
        self.discriminator.loss = lambda real_pred, fake_pred: (
            tf.keras.losses.MeanSquaredError()
            (real_pred, tf.ones_like(real_pred)) +
            tf.keras.losses.MeanSquaredError()
            (fake_pred, -tf.ones_like(fake_pred))
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """
        Generate a batch of fake samples via the generator.

        Parameters
        ----------
        size : int, optional
            Number of fake samples to generate. Defaults to self.batch_size.
        training : bool, optional
            If True, runs the generator in training mode
            (e.g. enables dropout).

        Returns
        -------
        tf.Tensor
            Generated fake samples.
        """
        if size is None:
            size = self.batch_size
        z = self.latent_generator(size)
        return self.generator(z, training=training)

    def get_real_sample(self, size=None):
        """
        Draw a random batch of real samples from the provided dataset.

        Parameters
        ----------
        size : int, optional
            Number of real samples to draw. Defaults to self.batch_size.

        Returns
        -------
        tf.Tensor
            Batch of real samples.
        """
        if size is None:
            size = self.batch_size
        indices = tf.range(tf.shape(self.real_examples)[0])
        shuffled = tf.random.shuffle(indices)[:size]
        return tf.gather(self.real_examples, shuffled)

    def train_step(self, data):
        """
        Perform one training step: update the discriminator for
        disc_iter steps, then update the generator once.

        Parameters
        ----------
        data : Ignored
            Not used. Required signature by Keras.

        Returns
        -------
        dict
            Dictionary containing 'discr_loss' and 'gen_loss'.
        """
        # ----- Update Discriminator -----
        for _ in range(self.disc_iter):
            real_batch = self.get_real_sample()
            fake_batch = self.get_fake_sample(training=True)
            with tf.GradientTape() as tape:
                real_pred = self.discriminator(real_batch, training=True)
                fake_pred = self.discriminator(fake_batch, training=True)
                d_loss = self.discriminator.loss(real_pred, fake_pred)
            grads = \
                tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        # ----- Update Generator -----
        with tf.GradientTape() as tape:
            fake_batch = self.get_fake_sample(training=True)
            pred = self.discriminator(fake_batch, training=False)
            g_loss = self.generator.loss(pred)
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        # Return losses for logging
        return {"discr_loss": d_loss, "gen_loss": g_loss}
