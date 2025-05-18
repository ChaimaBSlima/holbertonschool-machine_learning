#!/usr/bin/env python3
""" Task 1: 1. N-gram BLEU score """
import numpy as np


def ngramify(corpus, n):
    """
    Convert a corpus of tokenized sentences into n-grams.

    This function takes a list of sentences (each sentence is a list of tokens)
    or a single sentence (list of tokens) and returns the corresponding n-grams
    for each sentence. Each n-gram is represented as a string of tokens joined
    by spaces.

    Parameters:
    -----------
    corpus : list of list of str or list of str
        The input text data. Can be either:
        - A list of sentences, where each sentence is a list of tokens (words).
        - A single sentence as a list of tokens.
    n : int
        The n-gram size to generate
        (e.g., 1 for unigrams, 2 for bigrams, etc.).

    Returns:
    --------
    list of list of str or list of str
        If input is a list of sentences, returns a list of lists,
        where each inner list contains the n-grams (as strings) for the
        corresponding sentence.
        If input is a single sentence, returns a list of n-grams for
        that sentence.
    """
    unlist = 0

    if type(corpus[0]) is not list:
        corpus = [corpus]
        unlist = 1

    new_corpus = []
    for line in corpus:
        new_line = []

        for gram in range(len(line) - n + 1):
            new_gram = ""

            for i in range(n):
                if i != 0:
                    new_gram += " "

                new_gram += line[gram + i]

            new_line.append(new_gram)
        new_corpus.append(new_line)

    if unlist:
        return new_corpus[0]
    return new_corpus


def uni_bleu(references, sentence):
    """
    Calculate a unigram BLEU-like score for a candidate sentence
    against reference sentences.

    This function computes a simplified BLEU score based on unigram overlap,
    including a brevity penalty.
    It counts the maximum number of times each word appears in any reference
    and clips the counts in the candidate sentence accordingly. Then it
    calculates precision and applies a brevity penalty based on the closest
    reference length.

    Parameters:
    -----------
    references : list of list of str
        A list of reference sentences,
        each reference is a list of tokens (words).
    sentence : list of str
        The candidate sentence as a list of tokens (words) to be scored.

    Returns:
    --------
    float
        The unigram BLEU score (precision * brevity penalty)
        of the candidate sentence.
    """
    sent_dict = {}
    for word in sentence:
        sent_dict[word] = sent_dict.get(word, 0) + 1
    max_dict = {}

    for reference in references:
        this_ref = {}
        for word in reference:
            this_ref[word] = this_ref.get(word, 0) + 1

        for word in this_ref:
            max_dict[word] = max(max_dict.get(word, 0), this_ref[word])

    in_ref = 0
    for word in sent_dict:
        in_ref += min(max_dict.get(word, 0), sent_dict[word])

    closest = np.argmin(np.abs([len(ref) - len(sentence)
                                for ref in references]))

    closest = len(references[closest])

    if len(sentence) >= closest:
        brevity = 1

    else:
        brevity = np.exp(1 - closest / len(sentence))

    return brevity * in_ref / len(sentence)
