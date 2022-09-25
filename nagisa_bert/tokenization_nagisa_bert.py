# This code was created with reference to tokenization_bert.py,
# tokenization_bert_japanese.py from transformers and
# tokenization_bert_sudachipy.py from SudachiTra.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py
# https://github.com/huggingface/transformers/blob/983e40ac3b2af68fd6c927dce09324d54d023e54/src/transformers/models/bert_japanese/tokenization_bert_japanese.py
# https://github.com/WorksApplications/SudachiTra/blob/dbcaf5c851fe817bead20acf3958e50c93b0118c/sudachitra/tokenization_bert_sudachipy.py
"""Tokenization classes for nagisa BERT."""

import os
import copy

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import nagisa

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.bert.tokenization_bert import WordpieceTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# TODO: set offifical URL
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "taishi-i/nagisa_bert": "",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "taishi-i/nagisa_bert": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "taishi-i/nagisa_bert": {
        "do_lower_case": False,
        "word_tokenizer_type": "nagisa",
        "subword_tokenizer_type": "wordpiece",
    },
}


def load_vocabulary(
    vocab_file: str = VOCAB_FILES_NAMES["vocab_file"],
) -> Dict[str, int]:
    """
    Loads a vocabulary file into a dictionary.
    Args:
        vocab_file (str): Vocabulary file path.
    Returns:
        Dict[str, int]: Dictionary of vocabulary and its index.
    """
    vocab = OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n").split("\t")[0]
        vocab[token] = index
    return vocab


class NagisaBertTokenizer(PreTrainedTokenizer):
    """
    Construct a BERT tokenizer for Japanese text, based on a NagisaTokenizer.

    Args:
        vocab_file (`str`):
            Path to a one-wordpiece-per-line vocabulary file.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether to lower case the input. Only has an effect when do_basic_tokenize=True.
        do_word_tokenize (`bool`, *optional*, defaults to `True`):
            Whether to do word tokenization.
        do_subword_tokenize (`bool`, *optional*, defaults to `True`):
            Whether to do subword tokenization.
        word_tokenizer_type (`str`, *optional*, defaults to `"nagisa"`):
            Type of word tokenizer.
        subword_tokenizer_type (`str`, *optional*, defaults to `"wordpiece"`):
            Type of subword tokenizer.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        nagisa_kwargs (`str`, *optional*):
            Dictionary passed to the `NaigsaTokenizer` constructor.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        do_word_tokenize=True,
        do_subword_tokenize=True,
        word_tokenizer_type="nagisa",
        subword_tokenizer_type="wordpiece",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        nagisa_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            do_word_tokenize=do_word_tokenize,
            do_subword_tokenize=do_subword_tokenize,
            word_tokenizer_type=word_tokenizer_type,
            subword_tokenizer_type=subword_tokenizer_type,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            nagisa_kwargs=nagisa_kwargs,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'.")

        self.vocab = load_vocabulary(vocab_file)
        self.ids_to_tokens = OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )

        self.do_word_tokenize = do_word_tokenize
        self.word_tokenizer_type = word_tokenizer_type
        self.lower_case = do_lower_case
        self.nagisa_kwargs = copy.deepcopy(nagisa_kwargs)

        if not do_word_tokenize:
            raise ValueError("`do_word_tokenize` must be True.")

        if word_tokenizer_type == "nagisa":
            self.word_tokenizer = NagisaTokenizer(
                do_lower_case=do_lower_case,
                **(nagisa_kwargs or {}),
            )
        else:
            raise ValueError(
                f"Invalid word_tokenizer_type '{word_tokenizer_type}' is specified."
            )

        self.do_subword_tokenize = do_subword_tokenize
        self.subword_tokenizer_type = subword_tokenizer_type
        if do_subword_tokenize:
            if subword_tokenizer_type == "wordpiece":
                self.subword_tokenizer = WordpieceTokenizer(
                    vocab=self.vocab, unk_token=self.unk_token
                )
            else:
                raise ValueError(
                    f"Invalid subword_tokenizer_type '{subword_tokenizer_type}' is specified."
                )

    @property
    def do_lower_case(self):
        return self.lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def __getstate__(self):
        state = dict(self.__dict__)
        if self.word_tokenizer_type == "nagisa":
            del state["word_tokenizer"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if self.word_tokenizer_type == "nagisa":
            self.word_tokenizer = NagisaTokenizer(
                do_lower_case=self.do_lower_case, **(self.nagisa_kwargs or {})
            )

    def _tokenize(self, text, **kwargs):
        tokens = self.word_tokenizer.tokenize(text)

        if self.do_subword_tokenize:
            split_tokens = [
                sub_token
                for token in tokens
                for sub_token in self.subword_tokenizer.tokenize(token)
            ]
        else:
            split_tokens = tokens

        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "")
                + VOCAB_FILES_NAMES["vocab_file"],
            )
        else:
            vocab_file = (
                filename_prefix + "-" if filename_prefix else ""
            ) + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # logger.warning(
                    #     f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                    #     " Please check that the vocabulary is not corrupted!"
                    # )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


class NagisaTokenizer:
    def __init__(self, do_lower_case=False):
        self.lower = do_lower_case

    def tokenize(self, text, **kwargs):
        tokens = nagisa.wakati(text, lower=self.lower)
        tokens = [token for token in tokens if token != "\u3000"]
        return tokens
