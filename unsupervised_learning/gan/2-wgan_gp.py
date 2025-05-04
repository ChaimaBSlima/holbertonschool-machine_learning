#!/usr/bin/env python3
""" Task 2: 2. Wasserstein GANs with gradient penalty """
import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
    Wasserstein GAN with Gradient Penalty.

    This model implements the WGAN-GP algorithm:
      - The critic (discriminator) approximates the Wasserstein
      distance between real and generated distributions.
      - A gradient penalty term enforces the 1-Lipschitz condition
        instead of weight clipping.
      - The generator maximizes the critic's output on fake samples.

    Attributes
    ----------
    generator : keras.Model
        Maps latent vectors to generated data samples.
    discriminator : keras.Model
        Critic network scoring real vs. fake.
    latent_generator : callable
        Generates latent vectors given a batch size.
    real_examples : tf.Tensor
        Dataset of real samples for training the critic.
    batch_size : int
        Number of samples processed per batch.
    disc_iter : int
        Number of critic updates per generator update.
    learning_rate : float
        Learning rate for both Adam optimizers.
    beta_1 : float
        Exponential decay for the 1st moment estimates in Adam.
    beta_2 : float
        Exponential decay for the 2nd moment estimates in Adam.
    lambda_gp : float
        Weight coefficient for the gradient penalty term.
    axis : tf.Tensor
        Axes over which to compute gradient norm (all but batch axis).
    scal_shape : tf.TensorShape
        Shape for uniform interpolation between real and fake samples.

    Methods
    -------
    get_fake_sample(size=None, training=False)
        Generate `size` fake samples via the generator.
    get_real_sample(size=None)
        Draw `size` real samples from `real_examples`.
    get_interpolated_sample(real, fake)
        Create random convex combinations between real and fake batches.
    gradient_penalty(interpolated)
        Compute E[(||∇D(interpolated)||₂ - 1)²] over the batch.
    train_step(data)
        One optimization step: update critic (with GP), then generator.
    """

    def __init__(self,
                 generator,
                 discriminator,
                 latent_generator,
                 real_examples,
                 batch_size=200,
                 disc_iter=2,
                 learning_rate=0.005,
                 lambda_gp=10.0):
        """
        Initialize the WGAN-GP model.

        Parameters
        ----------
        generator : keras.Model
            Generator network.
        discriminator : keras.Model
            Critic network.
        latent_generator : callable
            Function mapping batch size to latent tensor of shape
            (batch_size, latent_dim).
        real_examples : tf.Tensor
            Tensor of real data with shape (N, ...).
        batch_size : int, optional
            Mini-batch size (default: 200).
        disc_iter : int, optional
            Number of critic updates per generator update (default: 2).
        learning_rate : float, optional
            Adam learning rate (default: 0.005).
        lambda_gp : float, optional
            Gradient penalty coefficient (default: 10.0).
        """
        super().__init__()
        # Core components
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        # Training hyperparameters
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = 0.3
        self.beta_2 = 0.9
        self.lambda_gp = lambda_gp

        # Precompute shapes for gradient penalty
        dims = tf.shape(self.real_examples)
        # axes = all dims except batch (axis 0)
        self.axis = tf.range(1, tf.rank(self.real_examples), dtype=tf.int32)
        # shape for mixing coefficient u: [batch_size] + [1,...,1]
        mix_shape =\
            [self.batch_size] + [1] * (self.real_examples.shape.rank - 1)
        self.scal_shape = mix_shape

        # Generator aims to maximize E[D(G(z))] => minimize -E[D(G(z))]
        self.generator.loss = lambda fake_pred: -tf.reduce_mean(fake_pred)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss)

        # Critic aims to minimize E[D(G(z))] - E[D(x)]
        self.discriminator.loss = lambda real_pred, fake_pred: (
            tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        Generate fake samples using the generator.
        """
        if size is None:
            size = self.batch_size
        z = self.latent_generator(size)
        return self.generator(z, training=training)

    def get_real_sample(self, size=None):
        """
        Draw a random batch of real samples.
        """
        if size is None:
            size = self.batch_size
        idx =\
            f.random.shuffle(tf.range(tf.shape(self.real_examples)[0]))[:size]
        return tf.gather(self.real_examples, idx)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Create interpolated samples:
        hat{x} = u*x_real + (1-u)*x_fake
        with u ~ Uniform[0,1] per sample.
        """
        u = tf.random.uniform(self.scal_shape, minval=0.0, maxval=1.0)
        return u * real_sample + (1.0 - u) * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Compute gradient penalty E[(||∇_x D(x)||₂ - 1)²
        on interpolated samples.
        """
        with tf.GradientTape() as tape:
            tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = tape.gradient(pred, interpolated_sample)
        # Compute L2 norm over non-batch axes
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        # Return mean penalty
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, data):
        """
        Execute one training iteration:
          1) Update critic disc_iter times with gradient penalty.
          2) Update generator once.

        Returns
        -------
        dict
            {'discr_loss': critic loss without GP,
             'gen_loss': generator loss,
             'gp': gradient_penalty value}
        """
        # Critic updates
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
                zip(grads, self.discriminator.trainable_variables))

        # Generator update
        with tf.GradientTape() as tape:
            fake_batch = self.get_fake_sample(training=True)
            fake_pred = self.discriminator(fake_batch, training=False)
            gen_loss = self.generator.loss(fake_pred)
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))

        return {'discr_loss': base_loss, 'gen_loss': gen_loss, 'gp': gp}
