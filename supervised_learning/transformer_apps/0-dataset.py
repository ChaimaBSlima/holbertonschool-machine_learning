#!/usr/bin/env python3
"""
Module that defines the Dataset class for loading and tokenizing
the TED Talks translation dataset (Portuguese to English).
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Loads and prepares a dataset for machine translation from
    Portuguese to English using the TED Talks dataset.
    """

    def __init__(self):
        """
        Class constructor. Loads the TED HRLR Portuguese to English dataset
        and tokenizes it using a custom vocabulary and Transformers tokenizer.
        """
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for both the Portuguese
        and English datasets.

        Args:
            data: tf.data.Dataset - Dataset containing (pt, en) sentence pairs.

        Returns:
            tokenizer_pt: PreTrainedTokenizerFast for Portuguese.
            tokenizer_en: PreTrainedTokenizerFast for English.
        """
        pt_corpus = []
        en_corpus = []

        for pt, en in data.take(10000):  # Limit for memory efficiency
            pt_corpus.append(pt.numpy().decode('utf-8'))
            en_corpus.append(en.numpy().decode('utf-8'))

        tokenizer_pt =\
            transformers.PreTrainedTokenizerFast.train_new_from_iterator(
                pt_corpus, vocab_size=2 ** 15
                )

        tokenizer_en =\
            transformers.PreTrainedTokenizerFast.train_new_from_iterator(
                en_corpus, vocab_size=2 ** 15
                )

        return tokenizer_pt, tokenizer_en
