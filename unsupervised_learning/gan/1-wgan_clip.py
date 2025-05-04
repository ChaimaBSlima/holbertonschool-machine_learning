#!/usr/bin/env python3
""" Task 1: 1. Wasserstein GANs """
import tensorflow as tf
from tensorflow import keras


class WGAN_clip(keras.Model):
    """
    Wasserstein GAN with weight clipping.

    Implements the WGAN training loop where the discriminator (critic) is
    trained to approximate the Wasserstein distance between real and generated
    distributions.
    Discriminator weights are clipped to enforce Lipschitz continuity.
    """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.005):
        """
        Initialize WGAN components and hyperparameters.

        Parameters
        ----------
        generator : keras.Model
            Network mapping latent vectors to generated samples.
        discriminator : keras.Model
            Critic network distinguishing real vs. fake samples.
        latent_generator : callable
            Function taking an integer batch size and returning latent tensors.
        real_examples : tf.Tensor
            Tensor of real data samples for critic training.
        batch_size : int, optional (default=200)
            Number of samples per training batch.
        disc_iter : int, optional (default=2)
            Number of critic updates per generator update.
        learning_rate : float, optional (default=0.005)
            Learning rate for both optimizers.
        """
        super().__init__()
        # Core models
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        # Training parameters
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # Generator: minimize -E[D(G(z))]
        self.generator.loss = lambda fake_pred: -tf.math.reduce_mean(fake_pred)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        # Discriminator: minimize E[D(G(z))] - E[D(x)]
        self.discriminator.loss = lambda real_pred, fake_pred: (
            tf.math.reduce_mean(fake_pred) - tf.math.reduce_mean(real_pred)
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
        Generate fake samples from the latent space.

        Parameters
        ----------
        size : int, optional
            Number of fake samples. Defaults to self.batch_size.
        training : bool, optional
            If True, runs the generator in training mode.

        Returns
        -------
        tf.Tensor
            Fake/generated data samples.
        """
        if size is None:
            size = self.batch_size
        z = self.latent_generator(size)
        return self.generator(z, training=training)

    def get_real_sample(self, size=None):
        """
        Sample a batch of real examples uniformly at random.

        Parameters
        ----------
        size : int, optional
            Number of real samples. Defaults to self.batch_size.

        Returns
        -------
        tf.Tensor
            Real data samples batch.
        """
        if size is None:
            size = self.batch_size
        idx = tf.range(tf.shape(self.real_examples)[0])
        idx = tf.random.shuffle(idx)[:size]
        return tf.gather(self.real_examples, idx)

    def train_step(self, data):
        """
        Perform one adversarial training step:
          1. Update discriminator (critic) disc_iter times:
             - Compute loss: E[D(G(z))] - E[D(x)]
             - Apply gradients and clip weights to [-1, 1]
          2. Update generator once:
             - Compute loss: -E[D(G(z))]
             - Apply gradients

        Parameters
        ----------
        data : Ignored
            Placeholder to match Keras train_step signature.

        Returns
        -------
        dict
            {'discr_loss': float, 'gen_loss': float}
        """
        # Critic updates
        for _ in range(self.disc_iter):
            real_batch = self.get_real_sample()
            fake_batch = self.get_fake_sample(training=True)
            with tf.GradientTape() as tape:
                real_pred = self.discriminator(real_batch, training=True)
                fake_pred = self.discriminator(fake_batch, training=True)
                d_loss = self.discriminator.loss(real_pred, fake_pred)
            grads =\
                tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )
            # Enforce Lipschitz constraint via weight clipping
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        # Generator update
        with tf.GradientTape() as tape:
            fake_batch = self.get_fake_sample(training=True)
            fake_pred = self.discriminator(fake_batch, training=False)
            g_loss = self.generator.loss(fake_pred)
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"discr_loss": d_loss, "gen_loss": g_loss}
