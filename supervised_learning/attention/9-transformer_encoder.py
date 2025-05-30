#!/usr/bin/env python3
""" Task 8: 8. Transformer Encoder """

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
        Transformer Encoder class.

    The encoder processes the input sequence by first applying an embedding
    and positional encoding, followed by a stack of encoder blocks that
    include multi-head self-attention and feed-forward layers.
    """

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
                Class constructor.

        Args:
            N (int): Number of encoder blocks to use.
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of units in the fully connected hidden layer.
            input_vocab (int): Size of the input vocabulary.
            max_seq_len (int): Maximum sequence length for positional encoding.
            drop_rate (float): Dropout rate.
        """
        super(Encoder, self).__init__()

        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Forward pass for the Encoder.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, input_seq_len).
            training (bool): Boolean indicating whether the model is in
                            training mode.
            mask (tf.Tensor): Mask to apply to the attention mechanism.

        Returns:
            tf.Tensor: Output of the encoder, shape
            (batch_size, input_seq_len, dm).
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x
