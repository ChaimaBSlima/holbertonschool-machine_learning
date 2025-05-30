#!/usr/bin/env python3
""" Task 1:  1. Self Attention """

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Custom implementation of Bahdanau-style self-attention mechanism.

    Reference: https://arxiv.org/pdf/1409.0473.pdf

    This layer computes a context vector for a sequence of hidden states
    given a previous decoder state (query).

    """

    def __init__(self, units):
        """
         Initializes the attention mechanism with three Dense layers.

        Args:
            units (int): Number of units in the dense layers used for
                        computing the attention scores.
        """

        # super() function that will make the child class inherit all the
        # methods and properties from its parent:

        super(SelfAttention, self).__init__()

        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.W = tf.keras.layers.Dense(units)

    def call(self, s_prev, hidden_states):
        """
        Computes the context vector and attention weights.

        Args:
            s_prev (Tensor): Previous decoder hidden state,
                            shape (batch, units).
            hidden_states (Tensor): Encoder hidden states, shape
                                    (batch, seq_len, units).

        Returns:
            Tuple[Tensor, Tensor]:
                - context: Weighted sum of hidden states (batch, units).
                - weight: Attention weights for each time step
                    (batch, seq_len, 1).
        """
        query = tf.expand_dims(s_prev, 1)

        # V - a Dense layer with 1 units, to be applied to the tanh of the
        # sum of the outputs of W and U

        tfadd = tf.math.add(self.W(query), self.U(hidden_states))
        score = self.V(tf.nn.tanh(tfadd))
        weight = tf.nn.softmax(score, axis=1)
        hs_we = weight * hidden_states
        context = tf.reduce_sum(hs_we, axis=1)

        return context, weight
