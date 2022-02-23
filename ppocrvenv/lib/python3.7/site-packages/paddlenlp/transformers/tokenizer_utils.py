# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import io
import json
import os
import six
import unicodedata
from shutil import copyfile
from typing import Iterable, Iterator, Optional, List, Any, Callable, Union

from paddle.utils import try_import
from paddlenlp.utils.downloader import get_path_from_url, COMMUNITY_MODEL_PREFIX
from paddlenlp.utils.env import MODEL_HOME
from paddlenlp.utils.log import logger
from dataclasses import dataclass, field

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

from ..data.vocab import Vocab
from .utils import InitTrackerMeta, fn_args_to_dict
from collections import OrderedDict

__all__ = [
    'PretrainedTokenizer', 'BPETokenizer', 'tokenize_chinese_chars',
    'is_chinese_char'
]


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming utf-8 input.
    Args:
        text (str|bytes): Text to be converted to unicode.
    Returns:
        str: converted text.
    """
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def whitespace_tokenize(text):
    """
    Runs basic whitespace cleaning and splitting on a peice of text.
    Args:
        text (str): Text to be tokened.
    Returns:
        list(str): Token list.
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_whitespace(char):
    """
    Checks whether `chars` is a whitespace character.
    """
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


@dataclass(frozen=True, eq=True)
class AddedToken:
    """
    AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
    way it should behave.
    """

    content: str = field(default_factory=str)
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = True

    def __getstate__(self):
        return self.__dict__


class Trie:
    """
    Trie in Python. Creates a Trie out of a list of words. The trie is used to split on `added_tokens` in one pass
    Loose reference https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self):
        self.data = {}

    def add(self, word: str):
        """
        Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
        The special key `""` is used to represent termination.

        This function is idempotent, adding twice the same word will leave the trie unchanged

        Example::

            >>> trie = Trie()
            >>> trie.add("Hello 友達")
            >>> trie.data
            {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}}
            >>> trie.add("Hello")
            >>> trie.data
            {"H": {"e": {"l": {"l": {"o": {"": 1, " ": {"友": {"達": {"": 1}}}}}}}}}
        """
        if not word:
            return
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[""] = 1

    def split(self, text: str) -> List[str]:
        """
        Will look for the words added to the trie within `text`. Output is the original string splitted along the
        boundaries of the words found.

        This trie will match the longest possible word first !

        Example::

            >>> trie = Trie()
            >>> trie.split("[CLS] This is a extra_id_100")
            ["[CLS] This is a extra_id_100"]
            >>> trie.add("[CLS]")
            >>> trie.add("extra_id_1")
            >>> trie.add("extra_id_100")
            >>> trie.split("[CLS] This is a extra_id_100")
            ["[CLS]", " This is a ", "extra_id_100"]
        """
        # indexes are counted left of the chars index.
        # "hello", index 0, is left of h, index 1 is between h and e.
        # index 5 is right of the "o".

        # States are going to capture every possible start (indexes as above)
        # as keys, and have as values, a pointer to the position in the trie
        # where we're at. This is a partial match for now.
        # This enables to keep track of multiple matches while we're iterating
        # the string
        # If the trie contains, "blowing", and "lower" and we encounter the
        # string "blower", we need to split into ["b", "lower"].
        # This is where we need to keep track of multiple possible starts.
        states = OrderedDict()

        # This will contain every indices where we need
        # to cut.
        # We force to cut at offset 0 and len(text) (added later)
        offsets = [0]

        # This is used by the lookahead which needs to skip over
        # some text where the full match exceeded the place in the initial
        # for loop
        skip = None
        # Main loop, Giving this algorithm O(n) complexity
        for current, current_char in enumerate(text):
            if skip and current < skip:
                # Prevents the lookahead for matching twice
                # like extra_id_100 and id_100
                continue

            # This will track every state
            # that stop matching, we need to stop tracking them.
            # If we look at "lowball", we're going to match "l" (add it to states), "o", "w", then
            # fail on "b", we need to remove 0 from the valid states.
            to_remove = set()
            # Whenever we found a match, we need to drop everything
            # this is a greedy algorithm, it will match on the first found token
            reset = False

            # In this case, we already have partial matches (But unfinished)
            for start, trie_pointer in states.items():
                if "" in trie_pointer:
                    # This is a final match, we need to reset and
                    # store the results in `offsets`.

                    # Lookahead to match longest first
                    # Important in case of extra_id_1 vs extra_id_100
                    # Here we are also actively looking for other earlier partial
                    # matches
                    # "[CLS]", "L", we need to match CLS even if L is special
                    for lookstart, looktrie_pointer in states.items():
                        if lookstart > start:
                            # This partial match is later, we can stop looking
                            break
                        elif lookstart < start:
                            # This partial match is earlier, the trie pointer
                            # was already updated, so index is + 1
                            lookahead_index = current + 1
                            end = current + 1
                        else:
                            # Here lookstart == start and
                            #      looktrie_pointer == trie_pointer
                            # It wasn't updated yet so indices are current ones
                            lookahead_index = current
                            end = current
                        next_char = text[
                            lookahead_index] if lookahead_index < len(
                                text) else None
                        while next_char in looktrie_pointer:
                            looktrie_pointer = looktrie_pointer[next_char]
                            lookahead_index += 1
                            if "" in looktrie_pointer:
                                start = lookstart
                                end = lookahead_index
                                skip = lookahead_index

                            if lookahead_index == len(text):
                                # End of string
                                break
                            next_char = text[lookahead_index]
                        # End lookahead

                        # Storing and resetting
                    offsets.append(start)
                    offsets.append(end)
                    reset = True
                    break
                elif current_char in trie_pointer:
                    # The current character being looked at has a match within the trie
                    # update the pointer (it will be stored back into states later).
                    trie_pointer = trie_pointer[current_char]

                    # Storing back the new pointer into the states.
                    # Partial matches got longer by one.
                    states[start] = trie_pointer
                else:
                    # The new character has not match in the trie, we need
                    # to stop keeping track of this partial match.
                    # We can't do it directly within the loop because of how
                    # python iteration works
                    to_remove.add(start)

            # Either clearing the full start (we found a real match)
            # Or clearing only the partial matches that didn't work.
            if reset:
                states = {}
            else:
                for start in to_remove:
                    del states[start]

            # If this character is a starting character within the trie
            # start keeping track of this partial match.
            if current_char in self.data:
                states[current] = self.data[current_char]

        # We have a cut at the end with states.
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                # This is a final match, we need to reset and
                # store the results in `offsets`.
                end = len(text)
                offsets.append(start)
                offsets.append(end)
                # Longest cut is always the one with lower start so the first
                # item so we need to break.
                break

        return self.cut_text(text, offsets)

    def cut_text(self, text, offsets):
        # We have all the offsets now, we just need to do the actual splitting.
        # We need to eventually add the first part of the string and the eventual
        # last part.
        offsets.append(len(text))
        tokens = []
        start = 0
        for end in offsets:
            if start > end:
                logger.error(
                    "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it anyway."
                )
                continue
            elif start == end:
                # This might happen if there's a match at index 0
                # we're also preventing zero-width cuts in case of two
                # consecutive matches
                continue
            tokens.append(text[start:end])
            start = end

        return tokens


def tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if is_chinese_char(cp):
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output


@six.add_metaclass(InitTrackerMeta)
class PretrainedTokenizer(object):
    """
    The base class for all pretrained tokenizers. It mainly provides common methods
    for loading (construction and loading) and saving pretrained tokenizers. Loading
    and saving also rely on the following class attributes which should be overridden
    by derived classes accordingly:

    - **tokenizer_config_file** (str): Represents the file name of tokenizer
      configuration for configuration saving and loading in local file system.
      The value is `tokenizer_config.json`.
    - **resource_files_names** (dict): Represents resources to specific file
      names mapping for resource saving and loading in local file system. The
      keys of dict representing resource items should be argument names in
      tokenizer's `__init__` method, and the values are file names for saving
      and loading corresponding resources. The mostly used resources here are
      vocabulary file and sentence-piece model file.
    - **pretrained_init_configuration** (dict): Provides the tokenizer configurations
      of built-in pretrained tokenizers (contrasts to tokenizers in local file
      system). It has pretrained tokenizer names as keys (the same as pretrained
      model names, such as `bert-base-uncased`), and the values are dict preserving
      corresponding configuration for tokenizer initialization.
    - **pretrained_resource_files_map** (dict): Provides resource URLs of built-in
      pretrained tokenizers (contrasts to tokenizers in local file system). It
      has the same keys as `resource_files_names`, and the values are also `dict`
      mapping specific pretrained tokenizer names (such as `bert-base-uncased`)
      to corresponding resource URLs.

    Moreover, methods common to tokenizers for tokenization, token/id conversion
    and encoding as model inputs are also provided here.

    Besides, metaclass `InitTrackerMeta` is used to create `PretrainedTokenizer`,
    by which subclasses can track arguments for initialization automatically
    and expose special tokens initialization used as attributes.
    """
    tokenizer_config_file = "tokenizer_config.json"
    pretrained_init_configuration = {}
    resource_files_names = {}  # keys are arguments of __init__
    pretrained_resource_files_map = {}
    padding_side = 'right'
    pad_token_type_id = 0
    special_tokens_map_extended = {}
    _additional_special_tokens = []

    def _wrap_init(self, original_init, *args, **kwargs):
        """
        It would be hooked after `__init__` to add specials tokens (arguments of
        `__init__` whose name ends with `_token`) as attributes of the tokenizer
        instance.
        """
        # expose tokens as attributes
        self.padding_side = kwargs.pop("padding_side", self.padding_side)
        assert self.padding_side in [
            "right", "left"
        ], "Padding side must be either left or right"
        init_dict = fn_args_to_dict(original_init, *args, **kwargs)
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}
        # TODO(guosheng): Use OrderedDict, otherwise `all_special_tokens` returns
        # a list without order.
        self.tokens_trie = Trie()
        self.special_tokens_map = {}
        for identifier, value in init_dict.items():
            if identifier.endswith('_token'):
                self.special_tokens_map[identifier] = value
            if identifier == "additional_special_tokens":
                assert isinstance(value, (
                    list, tuple)), f"Value {value} is not a list or tuple"
                self._additional_special_tokens += value
                assert all(
                    isinstance(t, (str, AddedToken)) for t in
                    value), "One of the tokens is not a string or an AddedToken"
                self.special_tokens_map[
                    identifier] = self._additional_special_tokens

        self.add_tokens(self.all_special_tokens, special_tokens=True)
        additional_special_tokens = []
        for token in self.all_special_tokens:
            if isinstance(token, AddedToken):
                token = token.content
            if token not in self.special_tokens_map.values():
                additional_special_tokens.append(token)
        self.special_tokens_map[
            "additional_special_tokens"] = additional_special_tokens

    def _build_special_tokens_map_extended(self, **kwargs):
        for identifier, token in kwargs.items():
            if identifier.endswith('_token') and isinstance(token, AddedToken):
                self.special_tokens_map_extended[identifier] = token

    def __call__(self,
                 text,
                 text_pair=None,
                 max_seq_len: Optional[int]=None,
                 stride=0,
                 is_split_into_words=False,
                 pad_to_max_seq_len=False,
                 truncation_strategy="longest_first",
                 return_position_ids=False,
                 return_token_type_ids=True,
                 return_attention_mask=False,
                 return_length=False,
                 return_overflowing_tokens=False,
                 return_special_tokens_mask=False):
        """
        Performs tokenization and uses the tokenized tokens to prepare model
        inputs. It supports sequence or sequence pair as input, and batch input
        is allowed. `self.encode()` or `self.batch_encode()` would be called
        separately for single or batch input depending on input format and
        `is_split_into_words` argument.

        Args:
            text (str, List[str] or List[List[str]]):
                The sequence or batch of sequences to be processed. One sequence
                is a string or a list of strings depending on whether it has been
                pretokenized. If each sequence is provided as a list of strings
                (pretokenized), you must set `is_split_into_words` as `True` to
                disambiguate with a batch of sequences.
            text_pair (str, List[str] or List[List[str]], optional):
                Same as `text` argument, while it represents for the latter
                sequence of the sequence pair.
            max_seq_len (int, optional):
                If set to a number, will limit the total sequence returned so
                that it has a maximum length. If there are overflowing tokens,
                those overflowing tokens will be added to the returned dictionary
                when `return_overflowing_tokens` is `True`. Defaults to `None`.
            stride (int, optional):
                Only available for batch input of sequence pair and mainly for
                question answering usage. When for QA, `text` represents questions
                and `text_pair` represents contexts. If `stride` is set to a
                positive number, the context will be split into multiple spans
                where `stride` defines the number of (tokenized) tokens to skip
                from the start of one span to get the next span, thus will produce
                a bigger batch than inputs to include all spans. Moreover, 'overflow_to_sample'
                and 'offset_mapping' preserving the original example and position
                information will be added to the returned dictionary. Defaults to 0.
            pad_to_max_seq_len (bool, optional):
                If set to `True`, the returned sequences would be padded up to
                `max_seq_len` specified length according to padding side
                (`self.padding_side`) and padding token id. Defaults to `False`.
            truncation_strategy (str, optional):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence
                until the input is under `max_seq_len` starting from the longest
                one at each token (when there is a pair of input sequences).
                - 'only_first': Only truncate the first sequence.
                - 'only_second': Only truncate the second sequence.
                - 'do_not_truncate': Do not truncate (raise an error if the input
                sequence is longer than `max_seq_len`).

                Defaults to 'longest_first'.
            return_position_ids (bool, optional):
                Whether to include tokens position ids in the returned dictionary.
                Defaults to `False`.
            return_token_type_ids (bool, optional):
                Whether to include token type ids in the returned dictionary.
                Defaults to `True`.
            return_attention_mask (bool, optional):
                Whether to include the attention mask in the returned dictionary.
                Defaults to `False`.
            return_length (bool, optional):
                Whether to include the length of each encoded inputs in the
                returned dictionary. Defaults to `False`.
            return_overflowing_tokens (bool, optional):
                Whether to include overflowing token information in the returned
                dictionary. Defaults to `False`.
            return_special_tokens_mask (bool, optional):
                Whether to include special tokens mask information in the returned
                dictionary. Defaults to `False`.

        Returns:
            dict or list[dict] (for batch input):
                The dict has the following optional items:

                - **input_ids** (list[int]): List of token ids to be fed to a model.
                - **position_ids** (list[int], optional): List of token position ids to be
                  fed to a model. Included when `return_position_ids` is `True`
                - **token_type_ids** (list[int], optional): List of token type ids to be
                  fed to a model. Included when `return_token_type_ids` is `True`.
                - **attention_mask** (list[int], optional): List of integers valued 0 or 1,
                  where 0 specifies paddings and should not be attended to by the
                  model. Included when `return_attention_mask` is `True`.
                - **seq_len** (int, optional): The input_ids length. Included when `return_length`
                  is `True`.
                - **overflowing_tokens** (list[int], optional): List of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **num_truncated_tokens** (int, optional): The number of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **special_tokens_mask** (list[int], optional): List of integers valued 0 or 1,
                  with 0 specifying special added tokens and 1 specifying sequence tokens.
                  Included when `return_special_tokens_mask` is `True`.
                - **offset_mapping** (list[int], optional): list of pair preserving the
                  index of start and end char in original input for each token.
                  For a special token, the index pair is `(0, 0)`. Included when
                  `stride` works.
                - **overflow_to_sample** (int, optional): Index of example from which this
                  feature is generated. Included when `stride` works.
        """
        # Input type checking for clearer error
        assert isinstance(text, str) or (
            isinstance(text, (list, tuple)) and (len(text) == 0 or (
                isinstance(text[0], str) or
                (isinstance(text[0], (list, tuple)) and
                 (len(text[0]) == 0 or isinstance(text[0][0], str)))))
        ), ("text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
            "or `List[List[str]]` (batch of pretokenized examples).")

        assert (text_pair is None or isinstance(text_pair, str) or (
            isinstance(text_pair, (list, tuple)) and (len(text_pair) == 0 or (
                isinstance(text_pair[0], str) or
                (isinstance(text_pair[0], (list, tuple)) and
                 (len(text_pair[0]) == 0 or isinstance(text_pair[0][0], str)))))
        )), (
            "text_pair input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
            "or `List[List[str]]` (batch of pretokenized examples).")

        is_batched = bool(
            (not is_split_into_words and isinstance(text, (list, tuple))) or
            (is_split_into_words and isinstance(text, (list, tuple)) and
             text and isinstance(text[0], (list, tuple))))

        if is_batched:
            batch_text_or_text_pairs = list(zip(
                text, text_pair)) if text_pair is not None else text
            return self.batch_encode(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                max_seq_len=max_seq_len,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_max_seq_len=pad_to_max_seq_len,
                truncation_strategy=truncation_strategy,
                return_position_ids=return_position_ids,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_length=return_length,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask)
        else:
            return self.encode(
                text=text,
                text_pair=text_pair,
                max_seq_len=max_seq_len,
                pad_to_max_seq_len=pad_to_max_seq_len,
                truncation_strategy=truncation_strategy,
                return_position_ids=return_position_ids,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_length=return_length,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask)

    @property
    def all_special_tokens(self):
        """ 
        list: All the special tokens ('<unk>', '<cls>'...) corresponding to
            special token arguments in `__init__` (arguments end with '_end').
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (
                list, tuple)) else [attr_value])
        all_toks = list(OrderedDict.fromkeys(all_toks))
        return all_toks

    @property
    def all_special_tokens_extended(self):
        """ 
        list: All the special tokens ('<unk>', '<cls>'...) corresponding to
            special token arguments in `__init__` (arguments end with '_end').
        """
        all_toks = []
        set_attr = self.special_tokens_map_extended
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (
                list, tuple)) else [attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    @property
    def all_special_ids(self):
        """ 
        list: All the token ids corresponding to all the special tokens.
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        return self.vocab_size + len(self.added_tokens_encoder)

    def _add_tokens(self, new_tokens, special_tokens=True):
        if special_tokens:
            add_special_tokens = []
            add_special_tokens_extended = []
            for token in new_tokens:
                if isinstance(token, AddedToken):
                    if token.content not in add_special_tokens:
                        self.tokens_trie.add(token.content)
                        add_special_tokens_extended.append(token)
                        add_special_tokens.append(token.content)
                        if token.content != self.unk_token and self.convert_tokens_to_ids(
                                token.content) == self.convert_tokens_to_ids(
                                    self.unk_token):
                            self.added_tokens_encoder[token.content] = len(self)
                            self.added_tokens_decoder[len(self) -
                                                      1] = token.content
                else:
                    if token not in add_special_tokens:
                        self.tokens_trie.add(token)
                        add_special_tokens.append(token)
                        if token != self.unk_token and self.convert_tokens_to_ids(
                                token) == self.convert_tokens_to_ids(
                                    self.unk_token):
                            self.added_tokens_encoder[token] = len(self)
                            self.added_tokens_decoder[len(self) - 1] = token

            self.special_tokens_map_extended[
                "additional_special_tokens"] = add_special_tokens_extended
        else:
            for token in new_tokens:
                if not isinstance(token, str):
                    raise TypeError(
                        f"Token {token} is not a string but a {type(token)}.")
                if hasattr(self, "do_lower_case") and self.do_lower_case:
                    token = token.lower()
                if token not in self.added_tokens_encoder and token != self.unk_token and self.convert_tokens_to_ids(
                        token) == self.convert_tokens_to_ids(self.unk_token):
                    self.added_tokens_encoder[token] = len(self)
                    self.added_tokens_decoder[len(self) - 1] = token

        return len(self.added_tokens_encoder)

    def add_tokens(self, new_tokens, special_tokens=True):
        if not new_tokens:
            return 0

        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]

        return self._add_tokens(new_tokens, special_tokens=special_tokens)

    def prepare_for_tokenization(self, text, **kwargs):
        return text

    def tokenize(self, text, **kwargs):
        all_special_tokens_extended = dict(
            (t.content, t) for t in self.all_special_tokens_extended
            if isinstance(t, AddedToken))
        no_split_token = set(self.all_special_tokens)
        text = self.prepare_for_tokenization(text, **kwargs)
        tokens = self.tokens_trie.split(text)
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = all_special_tokens_extended.get(token, None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        tokens[i + 1] = right.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()  # Opposite here
                else:
                    # We strip left and right by default
                    if right:
                        tokens[i + 1] = right.lstrip()
                    if left:
                        tokens[i - 1] = left.rstrip()
        tokenized_text = []
        for token in tokens:
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token, **kwargs))
        return tokenized_text

    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None
        if isinstance(tokens, str):
            if tokens in self.added_tokens_encoder:
                return self.added_tokens_encoder[tokens]
            else:
                return self._convert_token_to_id(tokens)
        ids = []
        for token in tokens:
            if token in self.added_tokens_encoder:
                ids.append(self.added_tokens_encoder[token])
            else:
                ids.append(self._convert_token_to_id(token))
        return ids

    def _convert_token_to_id(self, token):

        return self.vocab.to_indices(token)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (list of string) to a single string by
        using ``' '.join(tokens)`` .

        Args:
            tokens (list[str]): A sequence of tokens.

        Returns:
            str: Converted string.
        """
        return " ".join(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index):

        return self.vocab.to_tokens(index)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Creates an instance of `PretrainedTokenizer`. Related resources are loaded
        by specifying name of a built-in pretrained model, or a community-contributed
        pretrained model, or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of built-in pretrained model
                - Name of a community-contributed pretrained model.
                - Local directory path which contains tokenizer related resources
                  and tokenizer config file ("tokenizer_config.json").
            *args (tuple): position arguments for model `__init__`. If provided,
                use these as position argument values for tokenizer initialization.
            **kwargs (dict): keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for tokenizer
                initialization.

        Returns:
            PretrainedTokenizer: An instance of `PretrainedTokenizer`.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertTokenizer

                # Name of built-in pretrained model
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

                # Name of community-contributed pretrained model
                tokenizer = BertTokenizer.from_pretrained('yingyibiao/bert-base-uncased-sst-2-finetuned')

                # Load from local directory path
                tokenizer = BertTokenizer.from_pretrained('./my_bert/')
        """
        pretrained_models = list(cls.pretrained_init_configuration.keys())
        vocab_files = {}
        init_configuration = {}
        # From built-in pretrained models
        if pretrained_model_name_or_path in pretrained_models:
            for file_id, map_list in cls.pretrained_resource_files_map.items():
                vocab_files[file_id] = map_list[pretrained_model_name_or_path]
            init_configuration = copy.deepcopy(
                cls.pretrained_init_configuration[
                    pretrained_model_name_or_path])
        # From local dir path
        elif os.path.isdir(pretrained_model_name_or_path):
            for file_id, file_name in cls.resource_files_names.items():
                full_file_name = os.path.join(pretrained_model_name_or_path,
                                              file_name)
                if os.path.isfile(full_file_name):
                    vocab_files[file_id] = full_file_name
            vocab_files["tokenizer_config_file"] = os.path.join(
                pretrained_model_name_or_path, cls.tokenizer_config_file)
        else:
            # Assuming from community-contributed pretrained models
            for file_id, file_name in cls.resource_files_names.items():
                full_file_name = os.path.join(COMMUNITY_MODEL_PREFIX,
                                              pretrained_model_name_or_path,
                                              file_name)
                vocab_files[file_id] = full_file_name
            vocab_files["tokenizer_config_file"] = os.path.join(
                COMMUNITY_MODEL_PREFIX, pretrained_model_name_or_path,
                cls.tokenizer_config_file)

        default_root = os.path.join(MODEL_HOME, pretrained_model_name_or_path)
        resolved_vocab_files = {}
        for file_id, file_path in vocab_files.items():
            if file_path is None or os.path.isfile(file_path):
                resolved_vocab_files[file_id] = file_path
                continue
            path = os.path.join(default_root, file_path.split('/')[-1])
            if os.path.exists(path):
                logger.info("Already cached %s" % path)
                resolved_vocab_files[file_id] = path
            else:
                logger.info("Downloading %s and saved to %s" %
                            (file_path, default_root))
                try:
                    resolved_vocab_files[file_id] = get_path_from_url(
                        file_path, default_root)
                except RuntimeError as err:
                    logger.error(err)
                    raise RuntimeError(
                        f"Can't load tokenizer for '{pretrained_model_name_or_path}'.\n"
                        f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                        "- a correct model-identifier of built-in pretrained models,\n"
                        "- or a correct model-identifier of community-contributed pretrained models,\n"
                        "- or the correct path to a directory containing relevant tokenizer files.\n"
                    )

        # Prepare tokenizer initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        tokenizer_config_file = resolved_vocab_files.pop(
            "tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            with io.open(tokenizer_config_file, encoding="utf-8") as f:
                init_kwargs = json.load(f)
        else:
            init_kwargs = init_configuration
        # position args are stored in kwargs, maybe better not include
        init_args = init_kwargs.pop("init_args", ())
        init_kwargs.pop("init_class", None)

        # Update with newly provided args and kwargs
        init_args = init_args if not args else args
        init_kwargs.update(kwargs)

        # Merge resolved_vocab_files arguments in init_kwargs if not including.
        # Maybe need more ways to load resources.
        for args_name, file_path in resolved_vocab_files.items():
            # when `pretrained_model_name_or_path` is a pretrained model name,
            # use pretrained_init_configuration as `init_kwargs` to init which
            # does not include the vocab file in it, thus add vocab file into
            # args.
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path
            # when `pretrained_model_name_or_path` is a pretrained model dir,
            # use tokenizer_config_file.json as `init_kwargs` to init which
            # does include a vocab file path in it. However, if the vocab file
            # path included in json does not exist, such as was deleted, to make
            # it still work, use the vocab file under this dir.
            elif not os.path.isfile(init_kwargs[args_name]) and os.path.isfile(
                    file_path):
                init_kwargs[args_name] = file_path
        # TODO(guosheng): avoid reduplication of position args and key word args
        tokenizer = cls(*init_args, **init_kwargs)

        return tokenizer

    def save_pretrained(self, save_directory):
        """
        Save tokenizer configuration and related resources to files under
        `save_directory`. The tokenizer configuration would be saved into
        `tokenizer_config_file` indicating file (thus `tokenizer_config.json`),
        and resources would be saved into `resource_files_names` indicating files
        by using `self.save_resources(save_directory)`.
        
        The `save_directory` can be used in `from_pretrained` as argument value
        of `pretrained_model_name_or_path` to re-load the tokenizer.

        Args:
            save_directory (str): Directory to save files into.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertTokenizer

                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                tokenizer.save_pretrained('trained_model')
                # reload from save_directory
                tokenizer = BertTokenizer.from_pretrained('trained_model')
        """
        assert not os.path.isfile(
            save_directory
        ), "Saving directory ({}) should be a directory, not a file".format(
            save_directory)
        os.makedirs(save_directory, exist_ok=True)

        tokenizer_config_file = os.path.join(save_directory,
                                             self.tokenizer_config_file)
        # init_config is set in metaclass created `__init__`,
        tokenizer_config = self.init_config
        with io.open(tokenizer_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tokenizer_config, ensure_ascii=False))

        self.save_resources(save_directory)

    def save_resources(self, save_directory):
        """
        Save tokenizer related resources to `resource_files_names` indicating
        files under `save_directory` by copying directly. Override it if necessary.

        Args:
            save_directory (str): Directory to save files into.
        """
        for name, file_name in self.resource_files_names.items():
            src_path = self.init_config[name]
            dst_path = os.path.join(save_directory, file_name)
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                copyfile(src_path, dst_path)

    @staticmethod
    def load_vocabulary(filepath,
                        unk_token=None,
                        pad_token=None,
                        bos_token=None,
                        eos_token=None,
                        **kwargs):
        """
        Instantiate an instance of `Vocab` from a file reserving all tokens
        by using `Vocab.from_dict`. The file contains a token per line, and the
        line number would be the index of corresponding token.

        Args:
            filepath (str): path of file to construct vocabulary.
            unk_token (str): special token for unknown token. If no need, it also
                could be `None`. Defaults to `None`.
            pad_token (str): special token for padding token. If no need, it also
                could be `None`. Defaults to `None`.
            bos_token (str): special token for bos token. If no need, it also
                could be `None`. Defaults to `None`.
            eos_token (str): special token for eos token. If no need, it also
                could be `None`. Defaults to `None`.
            **kwargs (dict): keyword arguments for `Vocab.from_dict`.

        Returns:
            Vocab: An instance of `Vocab`.
        """
        token_to_idx = {}
        with io.open(filepath, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                token = line.rstrip('\n')
                token_to_idx[token] = int(index)
        vocab = Vocab.from_dict(
            token_to_idx,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs)
        return vocab

    @staticmethod
    def save_vocabulary(filepath, vocab):
        """
        Save all tokens to a vocabulary file. The file contains a token per line,
        and the line number would be the index of corresponding token.

        Args:
            filepath (str): File path to be saved to.
            vocab (Vocab|dict): The `Vocab` or `dict` instance to be saved.
        """
        if isinstance(vocab, Vocab):
            tokens = vocab.idx_to_token
        else:
            tokens = sorted(vocab.keys(), key=lambda token: vocab[token])
        with io.open(filepath, 'w', encoding='utf-8') as f:
            for token in tokens:
                f.write(token + '\n')

    def __getattr__(self, name):
        if name.endswith('_token'):
            return self.special_tokens_map[name]
        elif name.endswith('_token_id'):
            return self._convert_token_to_id(self.special_tokens_map[name[:-3]])
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def truncate_sequences(self,
                           ids,
                           pair_ids=None,
                           num_tokens_to_remove=0,
                           truncation_strategy='longest_first',
                           stride=0):
        """
        Truncates a sequence pair in place to the maximum length.

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            num_tokens_to_remove (:obj:`int`, `optional`, defaults to ``0``):
                number of tokens to remove using the truncation strategy
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_seq_len
                    starting from the longest one at each token (when there is a pair of input sequences).
                    Overflowing tokens only contains overflow from the first sequence.
                - 'only_first': Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_seq_len)
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_seq_len, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if truncation_strategy == 'longest_first':
            overflowing_tokens = []
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    overflowing_tokens = [ids[-1]] + overflowing_tokens
                    ids = ids[:-1]
                else:
                    pair_ids = pair_ids[:-1]
            window_len = min(len(ids), stride)
            if window_len > 0:
                overflowing_tokens = ids[-window_len:] + overflowing_tokens
        elif truncation_strategy == 'only_first':
            assert len(ids) > num_tokens_to_remove
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
        elif truncation_strategy == 'only_second':
            assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
        elif truncation_strategy == 'do_not_truncate':
            raise ValueError(
                "Input sequence are too long for max_length. Please select a truncation strategy."
            )
        else:
            raise ValueError(
                "Truncation_strategy should be selected in ['longest_first', 'only_first', 'only_second', 'do_not_truncate']"
            )
        return (ids, pair_ids, overflowing_tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            List[int]: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0

        return token_ids_0 + token_ids_1

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        """
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens.

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            offset_mapping_0 (List[tuple]):
                List of char offsets to which the special tokens will be added.
            offset_mapping_1 (List[tuple], optional):
                Optional second list of char offsets for offset mapping pairs.

        Returns:
            List[tuple]: List of char offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return offset_mapping_0

        return offset_mapping_0 + offset_mapping_1

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``encode`` methods.

        Args:
            token_ids_0 (List[int]): List of ids of the first sequence.
            token_ids_1 (List[int], optional): List of ids of the second sequence.
            already_has_special_tokens (bool, optional): Whether or not the token list is already
                formatted with special tokens for the model. Defaults to None.

        Returns:
            results (List[int]): The list of integers in the range [0, 1]:
                1 for a special token, 0 for a sequence token.
        """
        return [0] * ((len(token_ids_1)
                       if token_ids_1 else 0) + len(token_ids_0))

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.

        Should be overridden in a subclass if the model has a special way of building those.


        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (List[int]):
                List of IDs.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.

        Returns:
            List[int]: List of token_type_id according to the given sequence(s).
        """
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def num_special_tokens_to_add(self, pair):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Args:
            pair (bool, optional):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence. Defaults to `False`.
        Returns:
            int: Number of special tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def encode(self,
               text,
               text_pair=None,
               max_seq_len=512,
               pad_to_max_seq_len=False,
               truncation_strategy="longest_first",
               return_position_ids=False,
               return_token_type_ids=True,
               return_attention_mask=False,
               return_length=False,
               return_overflowing_tokens=False,
               return_special_tokens_mask=False):
        """
        Performs tokenization and uses the tokenized tokens to prepare model
        inputs. It supports sequence or sequence pair as input, and batch input
        is not allowed.

        Args:
            text (str, List[str] or List[int]):
                The sequence to be processed. One sequence is a string, a list
                of strings, or a list of integers depending on whether it has
                been pretokenized and converted to ids. 
            text_pair (str, List[str] or List[List[str]]):
                Same as `text` argument, while it represents for the latter
                sequence of the sequence pair.
            max_seq_len (int, optional):
                If set to a number, will limit the total sequence returned so
                that it has a maximum length. If there are overflowing tokens,
                those overflowing tokens will be added to the returned dictionary
                when `return_overflowing_tokens` is `True`. Defaults to `None`.
            stride (int, optional):
                Only available for batch input of sequence pair and mainly for
                question answering usage. When for QA, `text` represents questions
                and `text_pair` represents contexts. If `stride` is set to a
                positive number, the context will be split into multiple spans
                where `stride` defines the number of (tokenized) tokens to skip
                from the start of one span to get the next span, thus will produce
                a bigger batch than inputs to include all spans. Moreover, 'overflow_to_sample'
                and 'offset_mapping' preserving the original example and position
                information will be added to the returned dictionary. Defaults to 0.
            pad_to_max_seq_len (bool, optional):
                If set to `True`, the returned sequences would be padded up to
                `max_seq_len` specified length according to padding side
                (`self.padding_side`) and padding token id. Defaults to `False`.
            truncation_strategy (str, optional):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence
                until the input is under `max_seq_len` starting from the longest
                one at each token (when there is a pair of input sequences).
                - 'only_first': Only truncate the first sequence.
                - 'only_second': Only truncate the second sequence.
                - 'do_not_truncate': Do not truncate (raise an error if the input
                sequence is longer than `max_seq_len`).

                Defaults to 'longest_first'.
            return_position_ids (bool, optional):
                Whether to include tokens position ids in the returned dictionary.
                Defaults to `False`.
            return_token_type_ids (bool, optional):
                Whether to include token type ids in the returned dictionary.
                Defaults to `True`.
            return_attention_mask (bool, optional):
                Whether to include the attention mask in the returned dictionary.
                Defaults to `False`.
            return_length (bool, optional):
                Whether to include the length of each encoded inputs in the
                returned dictionary. Defaults to `False`.
            return_overflowing_tokens (bool, optional):
                Whether to include overflowing token information in the returned
                dictionary. Defaults to `False`.
            return_special_tokens_mask (bool, optional):
                Whether to include special tokens mask information in the returned
                dictionary. Defaults to `False`.

        Returns:
            dict:
                The dict has the following optional items:

                - **input_ids** (list[int]): List of token ids to be fed to a model.
                - **position_ids** (list[int], optional): List of token position ids to be
                  fed to a model. Included when `return_position_ids` is `True`
                - **token_type_ids** (list[int], optional): List of token type ids to be
                  fed to a model. Included when `return_token_type_ids` is `True`.
                - **attention_mask** (list[int], optional): List of integers valued 0 or 1,
                  where 0 specifies paddings and should not be attended to by the
                  model. Included when `return_attention_mask` is `True`.
                - **seq_len** (int, optional): The input_ids length. Included when `return_length`
                  is `True`.
                - **overflowing_tokens** (list[int], optional): List of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **num_truncated_tokens** (int, optional): The number of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **special_tokens_mask** (list[int], optional): List of integers valued 0 or 1,
                  with 0 specifying special added tokens and 1 specifying sequence tokens.
                  Included when `return_special_tokens_mask` is `True`.
        """

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        ids = get_input_ids(text)
        pair_ids = get_input_ids(text_pair) if text_pair is not None else None

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}

        # Truncation: Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(
            pair=pair))
        if max_seq_len and total_len > max_seq_len:

            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_seq_len,
                truncation_strategy=truncation_strategy, )
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_seq_len

        # Add special tokens

        sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
        token_type_ids = self.create_token_type_ids_from_sequences(ids,
                                                                   pair_ids)

        # Build output dictionnary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            encoded_inputs[
                "special_tokens_mask"] = self.get_special_tokens_mask(ids,
                                                                      pair_ids)
        if return_length:
            encoded_inputs["seq_len"] = len(encoded_inputs["input_ids"])

        # Check lengths
        assert max_seq_len is None or len(encoded_inputs[
            "input_ids"]) <= max_seq_len

        # Padding
        needs_to_be_padded = pad_to_max_seq_len and \
                             max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

        if needs_to_be_padded:
            difference = max_seq_len - len(encoded_inputs["input_ids"])
            if self.padding_side == 'right':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                        "input_ids"]) + [0] * difference
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] +
                        [self.pad_token_type_id] * difference)
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs[
                        "special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs[
                    "input_ids"] + [self.pad_token_id] * difference
            elif self.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [
                        1
                    ] * len(encoded_inputs["input_ids"])
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        [self.pad_token_type_id] * difference +
                        encoded_inputs["token_type_ids"])
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [
                    self.pad_token_id
                ] * difference + encoded_inputs["input_ids"]
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                    "input_ids"])

        if return_position_ids:
            encoded_inputs["position_ids"] = list(
                range(len(encoded_inputs["input_ids"])))

        return encoded_inputs

    def batch_encode(self,
                     batch_text_or_text_pairs,
                     max_seq_len=512,
                     pad_to_max_seq_len=False,
                     stride=0,
                     is_split_into_words=False,
                     truncation_strategy="longest_first",
                     return_position_ids=False,
                     return_token_type_ids=True,
                     return_attention_mask=False,
                     return_length=False,
                     return_overflowing_tokens=False,
                     return_special_tokens_mask=False):
        """
        Performs tokenization and uses the tokenized tokens to prepare model
        inputs. It supports batch inputs of sequence or sequence pair.

        Args:
            batch_text_or_text_pairs (list):
                The element of list can be sequence or sequence pair, and the
                sequence is a string or a list of strings depending on whether
                it has been pretokenized. If each sequence is provided as a list
                of strings (pretokenized), you must set `is_split_into_words` as
                `True` to disambiguate with a sequence pair.
            max_seq_len (int, optional):
                If set to a number, will limit the total sequence returned so
                that it has a maximum length. If there are overflowing tokens,
                those overflowing tokens will be added to the returned dictionary
                when `return_overflowing_tokens` is `True`. Defaults to `None`.
            stride (int, optional):
                Only available for batch input of sequence pair and mainly for
                question answering usage. When for QA, `text` represents questions
                and `text_pair` represents contexts. If `stride` is set to a
                positive number, the context will be split into multiple spans
                where `stride` defines the number of (tokenized) tokens to skip
                from the start of one span to get the next span, thus will produce
                a bigger batch than inputs to include all spans. Moreover, 'overflow_to_sample'
                and 'offset_mapping' preserving the original example and position
                information will be added to the returned dictionary. Defaults to 0.
            pad_to_max_seq_len (bool, optional):
                If set to `True`, the returned sequences would be padded up to
                `max_seq_len` specified length according to padding side
                (`self.padding_side`) and padding token id. Defaults to `False`.
            truncation_strategy (str, optional):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence
                until the input is under `max_seq_len` starting from the longest
                one at each token (when there is a pair of input sequences).
                - 'only_first': Only truncate the first sequence.
                - 'only_second': Only truncate the second sequence.
                - 'do_not_truncate': Do not truncate (raise an error if the input
                sequence is longer than `max_seq_len`).

                Defaults to 'longest_first'.
            return_position_ids (bool, optional):
                Whether to include tokens position ids in the returned dictionary.
                Defaults to `False`.
            return_token_type_ids (bool, optional):
                Whether to include token type ids in the returned dictionary.
                Defaults to `True`.
            return_attention_mask (bool, optional):
                Whether to include the attention mask in the returned dictionary.
                Defaults to `False`.
            return_length (bool, optional):
                Whether to include the length of each encoded inputs in the
                returned dictionary. Defaults to `False`.
            return_overflowing_tokens (bool, optional):
                Whether to include overflowing token information in the returned
                dictionary. Defaults to `False`.
            return_special_tokens_mask (bool, optional):
                Whether to include special tokens mask information in the returned
                dictionary. Defaults to `False`.

        Returns:
            list[dict]:
                The dict has the following optional items:

                - **input_ids** (list[int]): List of token ids to be fed to a model.
                - **position_ids** (list[int], optional): List of token position ids to be
                  fed to a model. Included when `return_position_ids` is `True`
                - **token_type_ids** (list[int], optional): List of token type ids to be
                  fed to a model. Included when `return_token_type_ids` is `True`.
                - **attention_mask** (list[int], optional): List of integers valued 0 or 1,
                  where 0 specifies paddings and should not be attended to by the
                  model. Included when `return_attention_mask` is `True`.
                - **seq_len** (int, optional): The input_ids length. Included when `return_length`
                  is `True`.
                - **overflowing_tokens** (list[int], optional): List of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **num_truncated_tokens** (int, optional): The number of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **special_tokens_mask** (list[int], optional): List of integers valued 0 or 1,
                  with 0 specifying special added tokens and 1 specifying sequence tokens.
                  Included when `return_special_tokens_mask` is `True`.
                - **offset_mapping** (list[int], optional): list of pair preserving the
                  index of start and end char in original input for each token.
                  For a sqecial token, the index pair is `(0, 0)`. Included when
                  `stride` works.
                - **overflow_to_sample** (int, optional): Index of example from which this
                  feature is generated. Included when `stride` works.
        """

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        batch_encode_inputs = []
        for example_id, tokens_or_pair_tokens in enumerate(
                batch_text_or_text_pairs):
            if not isinstance(tokens_or_pair_tokens, (list, tuple)):
                text, text_pair = tokens_or_pair_tokens, None
            elif is_split_into_words and not isinstance(
                    tokens_or_pair_tokens[0], (list, tuple)):
                text, text_pair = tokens_or_pair_tokens, None
            else:
                text, text_pair = tokens_or_pair_tokens

            first_ids = get_input_ids(text)
            second_ids = get_input_ids(
                text_pair) if text_pair is not None else None

            if stride > 0 and second_ids is not None:

                max_len_for_pair = max_seq_len - len(
                    first_ids) - self.num_special_tokens_to_add(pair=True)

                token_offset_mapping = self.get_offset_mapping(text)
                token_pair_offset_mapping = self.get_offset_mapping(text_pair)

                offset = 0
                while offset < len(second_ids):
                    encoded_inputs = {}
                    length = len(second_ids) - offset
                    if length > max_len_for_pair:
                        length = max_len_for_pair

                    ids = first_ids
                    pair_ids = second_ids[offset:offset + length]

                    mapping = token_offset_mapping
                    pair_mapping = token_pair_offset_mapping[offset:offset +
                                                             length]

                    offset_mapping = self.build_offset_mapping_with_special_tokens(
                        mapping, pair_mapping)
                    sequence = self.build_inputs_with_special_tokens(ids,
                                                                     pair_ids)
                    token_type_ids = self.create_token_type_ids_from_sequences(
                        ids, pair_ids)

                    # Build output dictionnary
                    encoded_inputs["input_ids"] = sequence
                    if return_token_type_ids:
                        encoded_inputs["token_type_ids"] = token_type_ids
                    if return_special_tokens_mask:
                        encoded_inputs[
                            "special_tokens_mask"] = self.get_special_tokens_mask(
                                ids, pair_ids)
                    if return_length:
                        encoded_inputs["seq_len"] = len(encoded_inputs[
                            "input_ids"])

                    # Check lengths
                    assert max_seq_len is None or len(encoded_inputs[
                        "input_ids"]) <= max_seq_len

                    # Padding
                    needs_to_be_padded = pad_to_max_seq_len and \
                                        max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

                    encoded_inputs['offset_mapping'] = offset_mapping

                    if needs_to_be_padded:
                        difference = max_seq_len - len(encoded_inputs[
                            "input_ids"])
                        if self.padding_side == 'right':
                            if return_attention_mask:
                                encoded_inputs["attention_mask"] = [1] * len(
                                    encoded_inputs[
                                        "input_ids"]) + [0] * difference
                            if return_token_type_ids:
                                # 0 for padding token mask
                                encoded_inputs["token_type_ids"] = (
                                    encoded_inputs["token_type_ids"] +
                                    [self.pad_token_type_id] * difference)
                            if return_special_tokens_mask:
                                encoded_inputs[
                                    "special_tokens_mask"] = encoded_inputs[
                                        "special_tokens_mask"] + [1
                                                                  ] * difference
                            encoded_inputs["input_ids"] = encoded_inputs[
                                "input_ids"] + [self.pad_token_id] * difference
                            encoded_inputs['offset_mapping'] = encoded_inputs[
                                'offset_mapping'] + [(0, 0)] * difference
                        elif self.padding_side == 'left':
                            if return_attention_mask:
                                encoded_inputs["attention_mask"] = [
                                    0
                                ] * difference + [1] * len(encoded_inputs[
                                    "input_ids"])
                            if return_token_type_ids:
                                # 0 for padding token mask
                                encoded_inputs["token_type_ids"] = (
                                    [self.pad_token_type_id] * difference +
                                    encoded_inputs["token_type_ids"])
                            if return_special_tokens_mask:
                                encoded_inputs["special_tokens_mask"] = [
                                    1
                                ] * difference + encoded_inputs[
                                    "special_tokens_mask"]
                            encoded_inputs["input_ids"] = [
                                self.pad_token_id
                            ] * difference + encoded_inputs["input_ids"]
                            encoded_inputs['offset_mapping'] = [
                                (0, 0)
                            ] * difference + encoded_inputs['offset_mapping']
                    else:
                        if return_attention_mask:
                            encoded_inputs["attention_mask"] = [1] * len(
                                encoded_inputs["input_ids"])

                    if return_position_ids:
                        encoded_inputs["position_ids"] = list(
                            range(len(encoded_inputs["input_ids"])))

                    encoded_inputs['overflow_to_sample'] = example_id
                    batch_encode_inputs.append(encoded_inputs)
                    if offset + length == len(second_ids):
                        break
                    offset += min(length, stride)

            else:
                batch_encode_inputs.append(
                    self.encode(
                        first_ids,
                        second_ids,
                        max_seq_len=max_seq_len,
                        pad_to_max_seq_len=pad_to_max_seq_len,
                        truncation_strategy=truncation_strategy,
                        return_position_ids=return_position_ids,
                        return_token_type_ids=return_token_type_ids,
                        return_attention_mask=return_attention_mask,
                        return_length=return_length,
                        return_overflowing_tokens=return_overflowing_tokens,
                        return_special_tokens_mask=return_special_tokens_mask))

        return batch_encode_inputs

    def get_offset_mapping(self, text):
        """
        Returns the map of tokens and the start and end index of their start and end character.
        Modified from https://github.com/bojone/bert4keras/blob/master/bert4keras/tokenizers.py#L372

        Args:
            text (str):
                Input text.
        Returns:
            list: The offset map of input text.
            
        """
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token
                                    if sub_token != self.unk_token else token)

        normalized_text, char_mapping = '', []

        for i, ch in enumerate(text):
            if self.basic_tokenizer.do_lower_case:
                ch = ch.lower()
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])

            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or _is_control(c))
            ])
            normalized_text += ch

            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0

        for token in split_tokens:
            if token[:2] == '##':
                token = token[2:]

            start = text[offset:].index(token) + offset
            end = start + len(token)

            token_mapping.append(
                (char_mapping[start], char_mapping[end - 1] + 1))
            offset = end

        return token_mapping


class BPETokenizer(PretrainedTokenizer):
    """
    The base class for all bpe tokenizers. It mainly provides common tokenize
    methods for bpe type tokenizer. 
    
    Args:
        vocab_file (str): 
            file path of the vocabulary.
        encoder_json_path (str, optional):
            file path of the id to vocab.
        vocab_bpe_path (str, optional):
            file path of word merge text.
        unk_token (str, optional): 
            The special token for unknown words. 
            Defaults to "[UNK]".
        sep_token (str, optional): 
            The special token for separator token. 
            Defaults to "[SEP]".
        pad_token (str, optional): 
            The special token for padding. 
            Defaults to "[PAD]".
        cls_token (str, optional): 
            The special token for cls. 
            Defaults to "[CLS]".
        mask_token (str, optional): 
            The special token for mask.
            Defaults to "[MASK]".

    """

    class Encoder(object):
        def __init__(self,
                     encoder,
                     bpe_merges,
                     errors='replace',
                     special_tokens=["[SEP]", "[p]", "[q]", "[/q]"]):
            self.encoder = encoder
            self.decoder = {v: k for k, v in self.encoder.items()}
            self.errors = errors  # how to handle errors in decoding
            self.byte_encoder = self._bytes_to_unicode()
            self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
            self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
            self.cache = {}
            self.re = try_import("regex")
            self.special_tokens = special_tokens

            # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
            self.pat = self.re.compile(
                r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            )

        @lru_cache()
        def _bytes_to_unicode(self):
            """
            Returns list of utf-8 byte and a corresponding list of unicode strings.
            The reversible bpe codes work on unicode strings.
            This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
            When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
            This is a signficant percentage of your normal, say, 32K bpe vocab.
            To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
            And avoids mapping to whitespace/control characters the bpe code barfs on.
            """

            bs = (list(range(ord("!"), ord("~") + 1)) +
                  list(range(ord("¡"), ord("¬") + 1)) +
                  list(range(ord("®"), ord("ÿ") + 1)))
            cs = bs[:]

            n = 0
            for b in range(2**8):
                if b not in bs:
                    bs.append(b)
                    cs.append(2**8 + n)
                    n += 1

            cs = [chr(n) for n in cs]

            ddict = dict(zip(bs, cs))
            return dict(zip(bs, cs))

        def _get_pairs(self, word):
            """Return set of symbol pairs in a word.
            Word is represented as tuple of symbols (symbols being variable-length strings).
            """
            pairs = set()
            prev_char = word[0]
            for char in word[1:]:
                pairs.add((prev_char, char))
                prev_char = char
            return pairs

        def bpe(self, token):
            if token in self.cache:
                return self.cache[token]
            word = tuple(token)
            pairs = self._get_pairs(word)

            if not pairs:
                return token

            while True:
                bigram = min(
                    pairs,
                    key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
                if bigram not in self.bpe_ranks:
                    break
                first, second = bigram
                new_word = []
                i = 0
                while i < len(word):
                    try:
                        j = word.index(first, i)
                        new_word.extend(word[i:j])
                        i = j
                    except:
                        new_word.extend(word[i:])
                        break

                    if word[i] == first and i < len(word) - 1 and word[
                            i + 1] == second:
                        new_word.append(first + second)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word = tuple(new_word)
                word = new_word
                if len(word) == 1:
                    break
                else:
                    pairs = self._get_pairs(word)
            word = ' '.join(word)
            self.cache[token] = word

            return word

        def tokenize(self, text):
            tokens = text.split(' ')
            sub_tokens = []
            for token_i, token in enumerate(tokens):
                if self.is_special_token(token):
                    if token_i == 0:
                        sub_tokens.extend([token])
                    else:
                        sub_tokens.extend([" " + token])
                else:
                    if token_i == 0:
                        sub_tokens.extend(self.re.findall(self.pat, token))
                    else:
                        sub_tokens.extend(
                            self.re.findall(self.pat, " " + token))
            return sub_tokens

        def tokenize_old(self, text):
            return self.re.findall(self.pat, text)

        def is_special_token(self, tok):
            if isinstance(tok, int):
                return False
            res = False
            for t in self.special_tokens:
                # if tok.find(t) != -1:
                if tok.strip() == t:
                    res = True
                    break
            return res

        def tokenize_bpe(self, token):

            if self.is_special_token(token):
                return [token.strip()]  # remove space for convert_to_ids
            else:

                token = ''.join(self.byte_encoder[b]
                                for b in token.encode('utf-8'))
                return [
                    self.encoder[bpe_token]
                    for bpe_token in self.bpe(token).split(' ')
                ]

        def encode(self, text):
            bpe_tokens = []
            for token in self.tokenize(text):
                bpe_tokens.extend(self.tokenize_bpe(token))
            return bpe_tokens

        def decode(self, tokens):
            pre_token_i = 0
            texts = []
            for token_i, token in enumerate(tokens):
                if self.is_special_token(token):
                    # proprecess tokens before token_i
                    if token_i - pre_token_i > 0:
                        text = ''.join([
                            self.decoder[int(tok)]
                            for tok in tokens[pre_token_i:token_i]
                        ])
                        text = bytearray(
                            [self.byte_decoder[c] for c in text]).decode(
                                'utf-8', errors=self.errors)
                        texts.append(text)
                    # texts.append(token)
                    if token_i == 0:
                        texts.append(
                            token
                        )  # in the beginning, there is no space before special tokens
                    else:
                        texts.extend(
                            [" ", token]
                        )  # in middle sentence, there must be a space before special tokens
                    pre_token_i = token_i + 1

            if pre_token_i < len(tokens):
                text = ''.join(
                    [self.decoder[int(tok)] for tok in tokens[pre_token_i:]])
                text = bytearray([self.byte_decoder[c] for c in text]).decode(
                    'utf-8', errors=self.errors)
                texts.append(text)

            return ''.join(texts)

    def __init__(self,
                 vocab_file,
                 encoder_json_path="./configs/encoder.json",
                 vocab_bpe_path="./configs/vocab.bpe",
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]"):
        self.vocab = self.load_vocabulary(
            vocab_file,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            mask_token=mask_token)
        self.encoder_json_path = encoder_json_path
        self.vocab_bpe_path = vocab_bpe_path
        self.encoder = self._get_encoder(encoder_json_path, vocab_bpe_path)
        self.nltk = try_import('nltk')

    def _tokenize(self, text, is_sentencepiece=True):
        text = convert_to_unicode(text)
        text = " ".join(text.split())  # remove duplicate whitespace
        if is_sentencepiece:
            sents = self.nltk.tokenize.sent_tokenize(text)
            bpe_ids = sum([self.encoder.encode(sent) for sent in sents], [])
        else:
            bpe_ids = self.encoder.encode(text)
        tokens = [str(bpe_id) for bpe_id in bpe_ids]
        return tokens

    def _get_encoder(self, encoder_json_path, vocab_bpe_path):
        with open(encoder_json_path, 'r') as f:
            encoder = json.load(f)
        with open(vocab_bpe_path, 'r') as f:
            bpe_data = f.read()
        bpe_merges = [
            tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]
        ]

        return self.Encoder(
            encoder=encoder,
            bpe_merges=bpe_merges, )
