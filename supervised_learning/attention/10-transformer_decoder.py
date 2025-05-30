#!/usr/bin/env python3
""" Task 9: 9. Transformer Decoder """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
     Transformer Decoder class.

    The decoder processes target sequences by applying embedding,
    positional encoding, and passing through a stack of decoder blocks.
    Each block performs masked self-attention, encoder-decoder attention,
    and feed-forward transformations.
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
                Class constructor.

        Args:
            N (int): Number of decoder blocks to use.
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of units in the feed-forward network.
            target_vocab (int): Size of the target vocabulary.
            max_seq_len (int): Maximum sequence length for positional encoding.
            drop_rate (float): Dropout rate.
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        self.blocks = [DecoderBlock(dm, h, hidden,
                                    drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
              Forward pass for the Decoder.

        Args:
            x (tf.Tensor): Input tensor of target tokens,
                            shape (batch_size, target_seq_len).
            encoder_output (tf.Tensor): Output from the encoder,
                            shape (batch_size, input_seq_len, dm).
            training (bool): Flag indicating if model is in
                                training mode (for dropout).
            look_ahead_mask (tf.Tensor): Mask to prevent attention
                                to future tokens in target sequences.
            padding_mask (tf.Tensor): Mask to ignore padding tokens
                                    in encoder output.

        Returns:
            tf.Tensor: Output tensor after decoder blocks, shape
            (batch_size, target_seq_len, dm).
        """
        seq_len = x.shape[1]
        x = self.embedding(x)
        x = x * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x = x + self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, encoder_output,
                               training, look_ahead_mask, padding_mask)

        return x
