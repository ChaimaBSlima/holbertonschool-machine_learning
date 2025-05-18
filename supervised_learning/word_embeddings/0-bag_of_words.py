#!/usr/bin/env python3
""" Task 0: 0. Bag Of Words"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Converts a list of sentences into a bag-of-words embedding matrix.

    Parameters:
        sentences (list of str): A list of input text strings.
        vocab (list of str, optional):
        A predefined vocabulary. If None, the vocabulary
        is determined from the sentences.

    Returns:
        tuple:
            - embedding (ndarray):
              A 2D NumPy array where each row represents the bag-of-words
              embedding of a sentence.
            - features (list of str):
                The list of feature names (i.e., vocabulary terms).
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    embedding = x.toarray()
    features = vectorizer.get_feature_names()

    return embedding, features
