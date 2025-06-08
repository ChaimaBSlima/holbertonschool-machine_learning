#!/usr/bin/env python3
""" Task 0: 0. Dataset """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Loads and prepares a dataset for machine translation from
    Portuguese to English using the TED Talks dataset.
    """

    def __init__(self):
        """
        Class constructor. Loads the TED HRLR Portuguese to English dataset
        and tokenizes it using SubwordTextEncoder.
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
            tokenizer_pt: SubwordTextEncoder for Portuguese.
            tokenizer_en: SubwordTextEncoder for English.
        """
        tokenizer_pt =\
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (pt.numpy() for pt, en in data),
                target_vocab_size=2 ** 15
                )

        tokenizer_en =\
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy() for pt, en in data),
                target_vocab_size=2 ** 15
                )

        return tokenizer_pt, tokenizer_en
