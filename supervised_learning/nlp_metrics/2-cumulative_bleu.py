#!/usr/bin/env python3
""" Task 2: 2. Cumulative N-gram BLEU score """
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculate the cumulative BLEU score up to n-grams for a candidate sentence.

    This function computes the BLEU score by combining modified
    n-gram precisionsfrom 1 to n, weighted equally, and applies a
    brevity penalty based on the closest reference length to the
    candidate sentence length.

    Parameters:
    -----------
    references : list of list of str
        A list of reference sentences, each represented as a list of tokens.
    sentence : list of str
        The candidate sentence as a list of tokens.
    n : int
        The maximum n-gram length to consider for the BLEU score.

    Returns:
    --------
    float
        The cumulative BLEU score (precision * brevity penalty)
        of the candidate sentence with respect to the references.
    """

    weight = 1 / n
    scores = [ngram_modscore(references, sentence, i, weight)
              for i in range(1, n + 1)]

    closest = np.argmin(np.abs([len(ref) - len(sentence)
                                for ref in references]))

    closest = len(references[closest])

    if len(sentence) >= closest:
        brevity = 1

    else:
        brevity = np.exp(1 - closest / len(sentence))

    return brevity * np.exp(sum(scores))


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


def ngram_modscore(references, sentence, n, weight):
    """
    Compute the weighted log of modified n-gram precision for a
    candidate sentence.

    This function calculates the modified n-gram precision
    by counting how many n-grams in the candidate sentence
    appear in the reference sentences, using clipped counts
    to avoid over-counting repeated n-grams. The result is
    then scaled by a weight and transformed using the natural
    logarithm, as part of the cumulative BLEU score calculation.

    Parameters:
    -----------
    references : list of list of str
        A list of reference sentences,
        where each reference is a list of tokens (words).
    sentence : list of str
        The candidate sentence as a list of tokens (words).
    n : int
        The n-gram size (e.g., 1 for unigrams, 2 for bigrams, etc.).
    weight : float
        The weight to apply to the log precision
        (typically 1/n when aggregating BLEU scores).

    Returns:
    --------
    float
        The weighted logarithm of the modified n-gram precision.
        Returns negative infinity if no n-gram matches (due to log(0),
        which is mathematically undefined).
    """
    references = ngramify(references, n)
    sentence = ngramify(sentence, n)
    sent_dict = {}

    for gram in sentence:
        sent_dict[gram] = sent_dict.get(gram, 0) + 1

    max_dict = {}
    for reference in references:
        this_ref = {}

        for gram in reference:
            this_ref[gram] = this_ref.get(gram, 0) + 1

        for gram in this_ref:
            max_dict[gram] = max(max_dict.get(gram, 0), this_ref[gram])

    in_ref = 0
    for gram in sent_dict:
        in_ref += min(max_dict.get(gram, 0), sent_dict[gram])

    return weight * np.log(in_ref / len(sentence))


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
