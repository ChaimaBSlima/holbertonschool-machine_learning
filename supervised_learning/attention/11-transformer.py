#!/usr/bin/env python3
""" Task 10: 10. Transformer Network """

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
        Transformer model class.

    This class encapsulates the Transformer architecture, composed of
    an encoder stack, a decoder stack, and a final linear layer that
    projects decoder outputs to the target vocabulary logits.
    """

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class constructor.

        Args:
            N (int): Number of layers in both encoder and decoder.
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of units in the feed-forward network.
            input_vocab (int): Size of the input vocabulary.
            target_vocab (int): Size of the target vocabulary.
            max_seq_input (int): Maximum input sequence length.
            max_seq_target (int): Maximum target sequence length.
            drop_rate (float): Dropout rate.
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(N, dm, h, hidden,
                               input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden,
                               target_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
                Forward pass for the Transformer model.

        Args:
            inputs (tf.Tensor):
            Input tensor of token indices (batch_size, input_seq_len).
            target (tf.Tensor):
            Target tensor of token indices (batch_size, target_seq_len).
            training (bool):
            Whether the model is in training mode (for dropout).
            encoder_mask (tf.Tensor):
            Mask to apply in the encoder self-attention.
            look_ahead_mask (tf.Tensor)
            Mask to prevent attention to future tokens in decoder.
            decoder_mask (tf.Tensor):
            Mask to apply in the encoder-decoder attention.

        Returns:
            tf.Tensor: Final output logits over the target vocabulary
                       (batch_size, target_seq_len, target_vocab).
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(
            target, enc_output, training, look_ahead_mask, decoder_mask)

        final_output = self.linear(dec_output)

        return final_output
