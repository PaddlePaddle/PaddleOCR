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
""" Tokenization classes for LayoutXLM model."""

import itertools
from dataclasses import dataclass, field
from collections import OrderedDict
from paddle.utils import try_import
from typing import List, Optional

from .. import PretrainedTokenizer, AddedToken
from ..tokenizer_utils import _is_punctuation, _is_control, _is_whitespace

SPIECE_UNDERLINE = "▁"


def _is_end_of_word(text):
    """Checks whether the last character in text is one of a punctuation, control or whitespace character."""
    last_char = text[-1]
    return bool(
        _is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(
            last_char))


def _is_start_of_word(text):
    """Checks whether the first character in text is one of a punctuation, control or whitespace character."""
    first_char = text[0]
    return bool(
        _is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(
            first_char))


class LayoutXLMTokenizer(PretrainedTokenizer):
    resource_files_names = {"vocab_file": "sentencepiece.bpe.model"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "layoutxlm-base-uncased":
            "https://bj.bcebos.com/paddlenlp/models/transformers/layoutxlm_base/sentencepiece.bpe.model",
        }
    }
    pretrained_init_configuration = {
        "layoutxlm-base-uncased": {
            "do_lower_case": False
        },
    }
    pretrained_positional_embedding_sizes = {"layoutxlm-base-uncased": 512, }
    max_model_input_sizes = pretrained_positional_embedding_sizes
    model_input_names = ["input_ids", "attention_mask"]

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]

    def __init__(self,
                 vocab_file,
                 bos_token="<s>",
                 eos_token="</s>",
                 sep_token="</s>",
                 cls_token="<s>",
                 unk_token="<unk>",
                 pad_token="<pad>",
                 mask_token="<mask>",
                 **kwargs):
        mask_token = AddedToken(
            mask_token, lstrip=True,
            rstrip=False) if isinstance(mask_token, str) else mask_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._sep_token = sep_token
        self._cls_token = cls_token
        self._unk_token = unk_token
        self._pad_token = pad_token
        self._mask_token = mask_token
        spm = try_import("sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        self.vocab_file = vocab_file

        self.tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # The first "real" token "," has position 4 in the original fairseq vocab and position 3 in the spm vocab
        self.offset = 1

        self.tokens_to_ids["<mask>"] = len(self.sp_model) + self.offset
        self.ids_to_tokens = {v: k for k, v in self.tokens_to_ids.items()}

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int],
            token_ids_1: Optional[List[int]]=None) -> List[int]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]]=None,
            already_has_special_tokens: bool=False) -> List[int]:
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)
                                                          ) + [1]

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int],
            token_ids_1: Optional[List[int]]=None) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        return len(self.sp_model) + self.offset + 1  # Add the <mask> token

    def get_vocab(self):
        vocab = {
            self.convert_ids_to_tokens(i): i
            for i in range(self.vocab_size)
        }
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.tokens_to_ids:
            return self.tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.ids_to_tokens:
            return self.ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))
