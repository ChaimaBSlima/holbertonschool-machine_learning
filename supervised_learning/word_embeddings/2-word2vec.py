#!/usr/bin/env python3
""" Task 2: 2. Train Word2Vec """
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Trains a Word2Vec model on the given sentences.

    Parameters:
        sentences (list of list of str):
            Tokenized text data, where each inner list
            represents a sentence as a list of words.

        size (int): Dimensionality of the word vectors.

        min_count (int):
        Ignores all words with total frequency lower than this.

        window (int):
        Maximum distance between the current and predicted
        word within a sentence.

        negative (int):
        Number of negative samples to use (for negative sampling).

        cbow (bool): If True, uses the Continuous Bag of Words architecture.
                     If False, uses the Skip-gram model.

        iterations (int): Number of iterations (epochs) over the corpus.

        seed (int): Seed for the random number generator.

        workers (int): Number of worker threads to train the model.

    Returns:
        gensim.models.Word2Vec: A trained Word2Vec model.
    """
    # Determine training algorithm: 0 = CBOW, 1 = Skip-gram
    sg = 0 if cbow else 1

    # Initialize model without training
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        seed=seed,
        workers=workers
    )

    # Build vocabulary from sentences
    model.build_vocab(sentences)

    # Train the model
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
