#!/usr/bin/env python3
"""Task 1:1. TF-IDF """
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Converts a list of sentences into a TF-IDF embedding matrix.

    Parameters:
        sentences (list of str): A list of input text strings.
        vocab (list of str, optional):
        A predefined vocabulary. If None, the vocabulary
        is learned from the sentences.

    Returns:
        tuple:
            - embedding (ndarray):
              A 2D NumPy array where each row represents the TF-IDF
              embedding of a sentence.
            - features (list of str):
            The list of feature names (i.e., vocabulary terms).
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    embedding = x.toarray()
    features = vectorizer.get_feature_names()

    return embedding, features
