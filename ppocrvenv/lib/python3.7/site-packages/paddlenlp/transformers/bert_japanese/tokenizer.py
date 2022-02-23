# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import unicodedata
import collections

from .. import BertTokenizer, BasicTokenizer, WordpieceTokenizer

__all__ = ['BertJapaneseTokenizer', 'MecabTokenizer', 'CharacterTokenizer']


class BertJapaneseTokenizer(BertTokenizer):
    """
    Construct a BERT tokenizer for Japanese text, based on a MecabTokenizer.

    Args:
        vocab_file (str):
            The vocabulary file path (ends with '.txt') required to instantiate
            a `WordpieceTokenizer`.
        do_lower_case (bool, optional):
            Whether or not to lowercase the input when tokenizing.
            Defaults to`False`.
        do_word_tokenize (bool, optional):
            Whether to do word tokenization. Defaults to`True`.
        do_subword_tokenize (bool, optional):
            Whether to do subword tokenization. Defaults to`True`.
        word_tokenizer_type (str, optional):
            Type of word tokenizer. Defaults to`basic`.
        subword_tokenizer_type (str, optional):
            Type of subword tokenizer. Defaults to`wordpiece`.
        never_split (bool, optional):
            Kept for backward compatibility purposes. Defaults to`None`.
        mecab_kwargs (str, optional):
            Dictionary passed to the `MecabTokenizer` constructor.
        unk_token (str):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to "[UNK]".
        sep_token (str):
            A special token separating two different sentences in the same input.
            Defaults to "[SEP]".
        pad_token (str):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to "[PAD]".
        cls_token (str):
            A special token used for sequence classification. It is the last token
            of the sequence when built with special tokens. Defaults to "[CLS]".
        mask_token (str):
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to "[MASK]".


    Examples:
        .. code-block::

            from paddlenlp.transformers import BertJapaneseTokenizer
            tokenizer = BertJapaneseTokenizer.from_pretrained('iverxin/bert-base-japanese/')

            inputs = tokenizer('こんにちは')
            print(inputs)

            '''
            {'input_ids': [2, 10350, 25746, 28450, 3], 'token_type_ids': [0, 0, 0, 0, 0]}
            '''

    """

    def __init__(self,
                 vocab_file,
                 do_lower_case=False,
                 do_word_tokenize=True,
                 do_subword_tokenize=True,
                 word_tokenizer_type="basic",
                 subword_tokenizer_type="wordpiece",
                 never_split=None,
                 mecab_kwargs=None,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 **kwargs):

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = BertJapaneseTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))

        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.idx_to_token.items()])

        self.do_word_tokenize = do_word_tokenize
        self.word_tokenizer_type = word_tokenizer_type
        self.lower_case = do_lower_case
        self.never_split = never_split
        self.mecab_kwargs = copy.deepcopy(mecab_kwargs)
        if do_word_tokenize:
            if word_tokenizer_type == "basic":
                self.basic_tokenizer = BasicTokenizer(
                    do_lower_case=do_lower_case, )
            elif word_tokenizer_type == "mecab":
                self.basic_tokenizer = MecabTokenizer(
                    do_lower_case=do_lower_case,
                    never_split=never_split,
                    **(mecab_kwargs or {}))
            else:
                raise ValueError(
                    f"Invalid word_tokenizer_type '{word_tokenizer_type}' is specified."
                )

        self.do_subword_tokenize = do_subword_tokenize
        self.subword_tokenizer_type = subword_tokenizer_type
        if do_subword_tokenize:
            if subword_tokenizer_type == "wordpiece":
                self.wordpiece_tokenizer = WordpieceTokenizer(
                    vocab=self.vocab, unk_token=unk_token)
            elif subword_tokenizer_type == "character":
                self.wordpiece_tokenizer = CharacterTokenizer(
                    vocab=self.vocab, unk_token=unk_token)
            else:
                raise ValueError(
                    f"Invalid subword_tokenizer_type '{subword_tokenizer_type}' is specified."
                )

    @property
    def do_lower_case(self):
        return self.lower_case

    def __getstate__(self):
        state = dict(self.__dict__)
        if self.word_tokenizer_type == "mecab":
            del state["basic_tokenizer"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if self.word_tokenizer_type == "mecab":
            self.basic_tokenizer = MecabTokenizer(
                do_lower_case=self.do_lower_case,
                never_split=self.never_split,
                **(self.mecab_kwargs or {}))

    def _tokenize(self, text):
        if self.do_word_tokenize:
            tokens = self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens)
        else:
            tokens = [text]

        if self.do_subword_tokenize:
            split_tokens = [
                sub_token
                for token in tokens
                for sub_token in self.wordpiece_tokenizer.tokenize(token)
            ]
        else:
            split_tokens = tokens

        return split_tokens


class MecabTokenizer:
    """Runs basic tokenization with MeCab morphological parser."""

    def __init__(
            self,
            do_lower_case=False,
            never_split=None,
            normalize_text=True,
            mecab_dic="ipadic",
            mecab_option=None, ):
        """
        Constructs a MecabTokenizer.

        Args:
            do_lower_case (bool): 
                Whether to lowercase the input. Defaults to`True`.
            never_split: (list): 
                Kept for backward compatibility purposes. Defaults to`None`.
            normalize_text (bool):
                Whether to apply unicode normalization to text before tokenization.  Defaults to`True`.
            mecab_dic (string):
                Name of dictionary to be used for MeCab initialization. If you are using a system-installed dictionary,
                set this option to `None` and modify `mecab_option`. Defaults to`ipadic`.
            mecab_option (string):
                String passed to MeCab constructor. Defaults to`None`.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else []
        self.normalize_text = normalize_text

        try:
            import fugashi
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install fugashi to use MecabTokenizer. "
                "See https://pypi.org/project/fugashi/ for installation.")

        mecab_option = mecab_option or ""

        if mecab_dic is not None:
            if mecab_dic == "ipadic":
                try:
                    import ipadic
                except ModuleNotFoundError as error:
                    raise error.__class__(
                        "The ipadic dictionary is not installed. "
                        "See https://github.com/polm/ipadic-py for installation."
                    )

                dic_dir = ipadic.DICDIR

            elif mecab_dic == "unidic_lite":
                try:
                    import unidic_lite
                except ModuleNotFoundError as error:
                    raise error.__class__(
                        "The unidic_lite dictionary is not installed. "
                        "See https://github.com/polm/unidic-lite for installation."
                    )

                dic_dir = unidic_lite.DICDIR

            elif mecab_dic == "unidic":
                try:
                    import unidic
                except ModuleNotFoundError as error:
                    raise error.__class__(
                        "The unidic dictionary is not installed. "
                        "See https://github.com/polm/unidic-py for installation."
                    )

                dic_dir = unidic.DICDIR
                if not os.path.isdir(dic_dir):
                    raise RuntimeError(
                        "The unidic dictionary itself is not found."
                        "See https://github.com/polm/unidic-py for installation."
                    )
            else:
                raise ValueError("Invalid mecab_dic is specified.")

            mecabrc = os.path.join(dic_dir, "mecabrc")
            mecab_option = f'-d "{dic_dir}" -r "{mecabrc}" ' + mecab_option

        self.mecab = fugashi.GenericTagger(mecab_option)

    def tokenize(self, text, never_split=None, **kwargs):
        """Tokenizes a piece of text."""
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        never_split = self.never_split + (never_split
                                          if never_split is not None else [])
        tokens = []

        for word in self.mecab(text):
            token = word.surface

            if self.do_lower_case and token not in never_split:
                token = token.lower()

            tokens.append(token)

        return tokens


class CharacterTokenizer:
    """Runs Character tokenization."""

    def __init__(self, vocab, unk_token, normalize_text=True):
        """
        Constructs a CharacterTokenizer.

        Args:
            vocab:
                Vocabulary object.
            unk_token (str):
                A special symbol for out-of-vocabulary token.
            normalize_text (boolean):
                Whether to apply unicode normalization to text before tokenization. Defaults to True.
        """
        self.vocab = vocab
        self.unk_token = unk_token
        self.normalize_text = normalize_text

    def tokenize(self, text):
        """
        Tokenizes a piece of text into characters.

        For example, `input = "apple""` wil return as output `["a", "p", "p", "l", "e"]`.

        Args:
            text: A single token or whitespace separated tokens.
                This should have already been passed through `BasicTokenizer`.

        Returns:
            A list of characters.
        """
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        output_tokens = []
        for char in text:
            if char not in self.vocab:
                output_tokens.append(self.unk_token)
                continue

            output_tokens.append(char)

        return output_tokens
