#!/usr/bin/env python3
""" Task 4: 4. Our own "This person does not exist" :
    Playing with a pre-trained model """
import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
    WGAN-GP with weight replacement.

    Inherits from keras.Model to train a Wasserstein GAN with gradient penalty.
    Adds:
      - replace_weights(gen_h5, disc_h5):
        load pretrained weights into sub-models.

    Parameters
    ----------
    generator : keras.Model
        Generator network mapping latent vectors to data.
    discriminator : keras.Model
        Critic network scoring real vs. fake samples.
    latent_generator : callable
        Function that generates latent input given batch size.
    real_examples : tf.Tensor
        Tensor of real data samples.
    batch_size : int
        Mini-batch size.
    disc_iter : int
        Critic updates per generator update.
    learning_rate : float
        Learning rate for Adam optimizers.
    lambda_gp : float
        Gradient penalty coefficient.

    Methods
    -------
    train_step(data)
        Performs one training iteration (critic with GP, then generator).
    replace_weights(gen_h5, disc_h5)
        Load weights from .h5 files into generator and discriminator.
    """
    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=0.005,
        lambda_gp=10.0
    ):
        super().__init__()
        # Core models
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        # Hyperparameters
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = 0.3
        self.beta_2 = 0.9
        self.lambda_gp = lambda_gp

        # Precompute axes and shape for gradient penalty
        self.axis = tf.range(1, tf.rank(self.real_examples), dtype=tf.int32)
        mix_shape =\
            [self.batch_size] + [1] * (self.real_examples.shape.rank - 1)
        self.scal_shape = mix_shape

        # Setup optimizer and loss for generator
        self.generator.loss = lambda fake_pred: -tf.reduce_mean(fake_pred)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        # Setup optimizer and loss for discriminator
        self.discriminator.loss = lambda real_pred, fake_pred: (
            tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
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
            Number of samples (default: batch_size).
        training : bool, optional
            If True, runs generator in training mode.

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
        Sample a batch of real examples.

        Parameters
        ----------
        size : int, optional
            Number of samples (default: batch_size).

        Returns
        -------
        tf.Tensor
            Real data batch.
        """
        if size is None:
            size = self.batch_size
        idx =\
            tf.random.shuffle(tf.range(tf.shape(self.real_examples)[0]))[:size]
        return tf.gather(self.real_examples, idx)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generate interpolated samples for gradient penalty.

        x_hat = u * real_sample + (1-u) * fake_sample with u ~ Uniform[0,1].
        """
        u = tf.random.uniform(self.scal_shape)
        return u * real_sample + (1.0 - u) * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Compute gradient penalty term: E[(||∇D(x_hat)||₂ - 1)²].
        """
        with tf.GradientTape() as tape:
            tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = tape.gradient(pred, interpolated_sample)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, data):
        """
        Execute one WGAN-GP training iteration.

        1) Update critic `disc_iter` times using gradient penalty.
        2) Update generator once.

        Returns
        -------
        dict
            'discr_loss': loss without GP,
            'gen_loss': generator loss,
            'gp': gradient penalty value
        """
        for _ in range(self.disc_iter):
            real_batch = self.get_real_sample()
            fake_batch = self.get_fake_sample(training=True)
            interpolated = self.get_interpolated_sample(real_batch, fake_batch)
            with tf.GradientTape() as tape:
                real_pred = self.discriminator(real_batch, training=True)
                fake_pred = self.discriminator(fake_batch, training=True)
                base_loss = self.discriminator.loss(real_pred, fake_pred)
                gp = self.gradient_penalty(interpolated)
                total_loss = base_loss + self.lambda_gp * gp
            grads =\
                tape.gradient(total_loss,
                              self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:
            fake_pred = self.discriminator(
                self.get_fake_sample(training=True), training=False
            )
            gen_loss = self.generator.loss(fake_pred)
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {'discr_loss': base_loss, 'gen_loss': gen_loss, 'gp': gp}

    def replace_weights(self, gen_h5, disc_h5):
        """
        Replace current model weights with pretrained weights.

        Parameters
        ----------
        gen_h5 : str
            Path to HDF5 file for generator weights.
        disc_h5 : str
            Path to HDF5 file for discriminator weights.
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)
