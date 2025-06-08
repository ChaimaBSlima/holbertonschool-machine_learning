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
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)
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
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased")

        def iterate_pt():
            """Generate Portuguese sentences one at a time from the dataset"""
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def iterate_en():
            """Generate English sentences one at a time from the dataset"""
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            iterate_pt(), vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            iterate_en(), vocab_size=2**13)
        return tokenizer_pt, tokenizer_en
