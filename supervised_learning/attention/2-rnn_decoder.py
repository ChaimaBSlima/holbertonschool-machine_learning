#!/usr/bin/env python3
""" Task 2: 2. RNN Decoder """

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Custom RNN decoder with attention mechanism.

    This decoder embeds input tokens, applies attention over encoder
    hidden states, and generates predictions using a GRU followed by
    a dense output layer.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes the decoder.

        Args:
            vocab (int): Size of the target vocabulary.
            embedding (int): Dimension of the embedding vectors.
            units (int): Number of GRU units.
            batch (int): Batch size.
        """
        super(RNNDecoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab, embedding)

        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
                Performs the forward pass of the decoder.

        Args:
            x (Tensor): Input token tensor of shape (batch, 1).
            s_prev (Tensor): Previous decoder hidden state of shape
                            (batch, units).
            hidden_states (Tensor): Encoder hidden states of shape
                                    (batch, seq_len, units).

        Returns:
            Tuple[Tensor, Tensor]:
                - y: Output logits for the next token prediction
                    (batch * 1, vocab).
                - s: Updated decoder hidden state (batch, units).
        """
        attention = SelfAttention(s_prev.shape[1])
        context, weights = attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        outputs, s = self.gru(x)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        y = self.F(outputs)

        return y, s
