#!/usr/bin/env python3
""" Task 3: 3. Extract Word2Vec """
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a trained Gensim Word2Vec model to a Keras Embedding layer.

    Parameters:
        model (gensim.models.Word2Vec): A trained Gensim Word2Vec model.

    Returns:
        keras.layers.Embedding: A Keras Embedding layer initialized with the
        weights from the Word2Vec model, trainable by default.
    """
    # Extract the weight matrix from the Word2Vec model
    weights = model.wv.vectors

    # Get vocabulary size and embedding dimensions
    vocab_size, embedding_dim = weights.shape

    # Create the TensorFlow Embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True
    )

    return embedding_layer
