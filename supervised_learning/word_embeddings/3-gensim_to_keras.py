#!/usr/bin/env python3
""" Task 3: 3. Extract Word2Vec """
import tensorflow.keras as keras

def gensim_to_keras(model):
    """
    Converts a trained Gensim Word2Vec model to a Keras Embedding layer.

    Parameters:
        model (gensim.models.Word2Vec): A trained Gensim Word2Vec model.

    Returns:
        keras.layers.Embedding: A Keras Embedding layer initialized with the
        weights from the Word2Vec model, trainable by default.
    """
    # Get weight matrix from Gensim Word2Vec model
    weights = model.wv.vectors

    # Get vocabulary size and embedding dimension
    vocab_size, vector_size = weights.shape

    # Create a Keras Embedding layer using these weights
    embedding_layer = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )

    return embedding_layer

