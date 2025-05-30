#!/usr/bin/env python3
""" Task 5: 5. Multi Head Attention"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Implements Multi-Head Attention as described in the Transformer
    architecture.

    Attributes:
        h (int): Number of attention heads.
        dm (int): Dimensionality of the model.
        depth (int): Depth of each attention head (dm / h).
        Wq (Dense): Dense layer to project the input query Q.
        Wk (Dense): Dense layer to project the input key K.
        Wv (Dense): Dense layer to project the input value V.
        linear (Dense): Final dense layer to combine the heads' outputs.
    """

    def __init__(self, dm, h):
        """
        Initializes the MultiHeadAttention layer.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = int(self.dm // self.h)
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def splitHeads(self, m, batch):
        """
        Splits the last dimension of a tensor into (h, depth)
        and transposes for attention.

        Args:
            m (tf.Tensor): Input tensor of shape (batch, seq_len, dm).
            batch (int): Batch size.

        Returns:
            tf.Tensor: Transformed tensor of shape (batch, h, seq_len, depth).
        """
        m = tf.reshape(m, (batch, -1, self.h, self.depth))
        return tf.transpose(m, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Applies the multi-head attention mechanism.

        Args:
            Q (tf.Tensor): Query tensor of shape (batch, seq_len_q, dm).
            K (tf.Tensor): Key tensor of shape (batch, seq_len_k, dm).
            V (tf.Tensor): Value tensor of shape (batch, seq_len_v, dm).
            mask (tf.Tensor or None): Optional mask tensor.

        Returns:
            tuple:
                - output (tf.Tensor): Output of shape (batch, seq_len_q, dm).
                - weights (tf.Tensor): Attention weights of shape
                    (batch, h, seq_len_q, seq_len_k).
        """
        batch = tf.shape(K)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        Q = self.splitHeads(Q, batch)
        K = self.splitHeads(K, batch)
        V = self.splitHeads(V, batch)
        output, weights = sdp_attention(Q, K, V, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch, -1, self.dm))
        output = self.linear(output)

        return output, weights
