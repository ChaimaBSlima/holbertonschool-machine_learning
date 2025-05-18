#!/usr/bin/env python3
""" Task 0:0. Unigram BLEU score """

import numpy as np


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
