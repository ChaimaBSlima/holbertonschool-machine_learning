#!/usr/bin/env python3
""" Task 3: 3. Extract Word2Vec """
from tensorflow import keras


def gensim_to_keras(model):
    """
    Converts a trained Gensim Word2Vec model to a Keras Embedding layer.

    Parameters:
        model (gensim.models.Word2Vec): A trained Gensim Word2Vec model.

    Returns:
        keras.layers.Embedding: A Keras Embedding layer initialized with the
        weights from the Word2Vec model, trainable by default.
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
