#!/usr/bin/env python3
""" Task 6: 6. Transformer Encoder Block """

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Represents a single encoder block for the Transformer model.

    This block includes:
    - Multi-head self-attention mechanism
    - Position-wise feed-forward network
    - Dropout and residual connections with layer normalization
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initializes the encoder block.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the feed-forward network.
            drop_rate (float): Dropout rate to apply after attention and
                            FFN layers.
        """
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Applies the encoder block to the input tensor.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, input_seq_len, dm).
            training (bool): Whether the model is in training mode
                            (for dropout).
            mask (tf.Tensor or None): Optional mask to prevent attention to
            certain positions.

        Returns:
            tf.Tensor: The output tensor after attention and feed-forward
            processing, same shape as input.
        """
        attnOutput, _ = self.mha(x, x, x, mask)
        attnOutput = self.dropout1(attnOutput, training=training)
        output_3 = self.layernorm1(x + attnOutput)
        output_2 = self.dense_hidden(output_3)
        output_1 = self.dense_output(output_2)
        output_0 = self.dropout2(output_1, training=training)
        output = self.layernorm2(output_3 + output_0)

        return output
