#!/usr/bin/env python3
""" Task 3: 3. Positional Encoding """


import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Generates a positional encoding matrix as described in the
    Transformer architecture.

    Positional encodings inject information about the relative or
    absolute positionnof tokens in a sequence. The encoding has the
    same dimension as the embeddingsnand is added to the input
    embeddings at the bottoms of the encoder and decoder stacks.

    Args:
        max_seq_len (int): Maximum length of the input sequences.
        dm (int): Dimensionality of the model (i.e., embedding size).

    Returns:
        numpy.ndarray: A matrix of shape (max_seq_len, dm)
        containing the positional encodings.
    Generates a positional encoding matrix as described in the
    Transformer architecture.

    Positional encodings inject information about the relative or
    absolute position of tokens in a sequence. The encoding has
    the same dimension as the embeddings and is added to the input
    embeddings at the bottoms of the encoder and decoder stacks.

    Args:
        max_seq_len (int): Maximum length of the input sequences.
        dm (int): Dimensionality of the model (i.e., embedding size).

    Returns:
        numpy.ndarray:
        A matrix of shape (max_seq_len, dm) containing the positional
        encodings.
    """

    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = 2 * (np.arange(dm)[np.newaxis, :]//2) / np.float32(dm)

    pev = pos / np.power(10000, i)

    # Applying SIN to odd indices
    pev[:, 0::2] = np.sin(pev[:, 0::2])

    # Applying COS to odd indices
    pev[:, 1::2] = np.cos(pev[:, 1::2])

    return pev
