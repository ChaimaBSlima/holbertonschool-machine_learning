#!/usr/bin/env python3
""" Task 7: 7. Transformer Decoder Block """

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Transformer decoder block that processes target input sequences.

    It includes:
    - Masked multi-head self-attention (to prevent access to future tokens)
    - Encoder-decoder multi-head attention (attending to encoder output)
    - Feed-forward network
    - Residual connections and normalization layers
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initializes the decoder block.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the feed-forward network.
            drop_rate (float): Dropout rate.
        """
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Applies the decoder block to the input.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, target_seq_len, dm).
            encoder_output (tf.Tensor): Encoder output of shape
                                        (batch, input_seq_len, dm).
            training (bool): Training mode flag (used for dropout).
            look_ahead_mask (tf.Tensor): Mask to prevent attention to future
                                        tokens.
            padding_mask (tf.Tensor): Mask to prevent attention to padding
                                    tokens in encoder output.

        Returns:
            tf.Tensor: The output of the decoder block with shape
            (batch, target_seq_len, dm).
        """
        attn_1, weights_b1 = self.mha1(x, x, x, look_ahead_mask)
        attn_1 = self.dropout1(attn_1, training=training)
        output1 = self.layernorm1(x + attn_1)

        attn_2, weights_b2 = self.mha2(output1,
                                       encoder_output,
                                       encoder_output,
                                       padding_mask)
        attn_2 = self.dropout2(attn_2, training=training)

        output2 = self.layernorm2(attn_2 + output1)
        ffn_out = self.dense_hidden(output2)
        ffn_out = self.dense_output(ffn_out)
        ffn_out = self.dropout3(ffn_out, training=training)
        output3 = self.layernorm3(ffn_out + output2)

        return output3
