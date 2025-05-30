#!/usr/bin/env python3
""" Task 0: 0. RNN Encoder """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    A custom RNN encoder class that uses GRU
    (Gated Recurrent Unit) for encoding input sequences.

    Attributes:
        batch (int): Batch size.
        units (int): Number of GRU units.

        embedding (tf.keras.layers.Embedding):
        Embedding layer for converting input tokens to dense vectors.

        gru (tf.keras.layers.GRU):
        GRU layer to process the embedded inputs.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor that initializes the embedding and GRU layers.

        Args:
            vocab (int): Size of the input vocabulary.
            embedding (int): Dimension of the embedding vectors.
            units (int): Number of units in the GRU layer.
            batch (int): Batch size.
        """
        # super() function that will make the child class inherit all the
        # methods and properties from its parent:

        init_weights = 'glorot_uniform'
        super(RNNEncoder, self).__init__()

        self.batch, self.units = batch, units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer=init_weights,
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        Initializes the hidden state of the GRU with zeros.

        Returns:
            Tensor:
            A tensor of zeros with shape (batch, units)
            to be used as the initial hidden state.
        """
        return tf.keras.initializers.Zeros()(shape=(self.batch, self.units))

    def call(self, x, initial):
        """
        Defines the forward pass of the encoder.

        Args:
            x (Tensor): Input tensor containing token indices.
            initial (Tensor): Initial hidden state for the GRU.

        Returns:
            Tuple[Tensor, Tensor]:
            The output sequences and the final hidden state.
        """
        outputs, hidden = self.gru(self.embedding(x), initial_state=initial)
        return outputs, hidden
