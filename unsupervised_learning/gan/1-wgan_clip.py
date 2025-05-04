#!/usr/bin/env python3
""" Task 1: 1. Wasserstein GANs """
import tensorflow as tf
from tensorflow import keras


class WGAN_clip(keras.Model):
    """
    Wasserstein GAN with weight clipping.

    Trains discriminator to approximate Wasserstein distance,
    clipping its weights to lie in [-1,1].
    """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.005):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # WGAN generator loss: minimize -E[D(G(z))]
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

        # WGAN discriminator loss: minimize E[D(G(z))] - E[D(x)]
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
        if size is None:
            size = self.batch_size
        z = self.latent_generator(size)
        return self.generator(z, training=training)

    def get_real_sample(self, size=None):
        if size is None:
            size = self.batch_size
        indices = tf.range(tf.shape(self.real_examples)[0])
        shuffled = tf.random.shuffle(indices)[:size]
        return tf.gather(self.real_examples, shuffled)

    def train_step(self, data):
        # Update discriminator disc_iter times
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
            # Clip discriminator weights to [-1, 1]
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        # Update generator once
        with tf.GradientTape() as tape:
            fake_batch = self.get_fake_sample(training=True)
            pred = self.discriminator(fake_batch, training=False)
            g_loss = self.generator.loss(pred)
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"discr_loss": d_loss, "gen_loss": g_loss}
