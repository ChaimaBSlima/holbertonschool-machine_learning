#!/usr/bin/env python3
""" Task 4: 4. Scaled Dot Product Attention """


import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the Scaled Dot-Product Attention.

    Scaled dot-product attention is computed using the following steps:
    1. Compute the dot product of the query (Q) and key (K^T).
    2. Scale the result by the square root of the dimensionality of the
        key vectors.
    3. Optionally apply a mask to prevent attention to certain positions.
    4. Apply softmax to obtain the attention weights.
    5. Multiply the weights by the value (V) vectors to get the output.

    Args:
        Q (tf.Tensor): Query tensor of shape (..., seq_len_q, depth).
        K (tf.Tensor): Key tensor of shape (..., seq_len_k, depth).
        V (tf.Tensor): Value tensor of shape (..., seq_len_v, depth_v),
                        usually seq_len_k == seq_len_v.
        mask (tf.Tensor, optional): Optional mask tensor broadcastable
        to (..., seq_len_q, seq_len_k).

    Returns:
        tuple: A tuple of two tensors:
            - output (tf.Tensor):
            Result of attention mechanism (..., seq_len_q, depth_v).
            - weights (tf.Tensor):
            Attention weights (..., seq_len_q, seq_len_k).
    """

    # scale matmul_qk with scaled_attention_logits
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    sal = matmul_qk / tf.math.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))

    # add the mask
    sal = sal + (mask * -1e9) if mask is not None else sal

    # softmax
    weights = tf.nn.softmax(sal, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
