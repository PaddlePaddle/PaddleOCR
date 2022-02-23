# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
"""Tokenization class for XLNet model."""

import os
import unicodedata
from shutil import copyfile
from typing import List, Optional

from paddle.utils import try_import
from .. import PretrainedTokenizer

__all__ = ['XLNetTokenizer']

SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # Kept for backward compatibility

# Segments (not really needed)
SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4


class XLNetTokenizer(PretrainedTokenizer):
    """
    Constructs an XLNet tokenizer based on `SentencePiece <https://github.com/google/sentencepiece>`__.

    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.

    Args:
        vocab_file (str):
            The vocabulary file (ends with '.spm') required to instantiate
            a `SentencePiece <https://github.com/google/sentencepiece>`__ tokenizer.
        do_lower_case (bool, optional):
            Whether or not to lowercase the input when tokenizing. Defaults to `False` and
            **does not** lowercase the input.
        remove_space (bool, optional):
            Whether or not to strip the text when tokenizing. Defaults to `True` and
            removes excess spaces before and after the string.
        keep_accents (bool, optional):
            Whether or not to keep accents when tokenizing. Defaults to `False` and **does not** keep accents.
        bos_token (str, optional):
            A special token representing the beginning of a sequence that was used during pretraining.
            Defaults to `"<s>"`.
        eos_token (str, optional):
            A special token representing the end of a sequence that was used during pretraining.
            Defaults to `"</s>"`.
        unk_token (str, optional):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to `"<unk>"`.
        sep_token (str, optional):
            A special token separating two different sentences in the same input.
            Defaults to `"<sep>"`.
        pad_token (str, optional):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to `"<pad>"`.
        cls_token (str, optional):
            A special token used for sequence classification. It is the last token
            of the sequence when built with special tokens. Defaults to `"<cls>"`.
        mask_token (str, optional):
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to `"<mask>"`.
        additional_special_tokens (List[str], optional):
            A list of additional special tokens to be used by the tokenizer.
            Defaults to `["<eop>", "<eod>"]`.

    Attributes:
        sp_model (SentencePieceProcessor):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    resource_files_names = {"vocab_file": "spiece.model"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "xlnet-base-cased":
            "https://bj.bcebos.com/paddlenlp/models/transformers/xlnet/xlnet-base-cased-spiece.model",
            "xlnet-large-cased":
            "https://bj.bcebos.com/paddlenlp/models/transformers/xlnet/xlnet-large-cased-spiece.model",
            "chinese-xlnet-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/xlnet/chinese-xlnet-base-spiece.model",
            "chinese-xlnet-mid":
            "https://bj.bcebos.com/paddlenlp/models/transformers/xlnet/chinese-xlnet-mid-spiece.model",
            "chinese-xlnet-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/xlnet/chinese-xlnet-large-spiece.model",
        }
    }
    pretrained_init_configuration = {
        "xlnet-base-cased": {
            "do_lower_case": False
        },
        "xlnet-large-cased": {
            "do_lower_case": False
        },
        "chinese-xlnet-base": {
            "do_lower_case": False
        },
        "chinese-xlnet-mid": {
            "do_lower_case": False
        },
        "chinese-xlnet-large": {
            "do_lower_case": False
        },
    }
    pretrained_positional_embedding_sizes = {
        "xlnet-base-cased": None,
        "xlnet-large-cased": None,
        "chinese-xlnet-base": None,
        "chinese-xlnet-mid": None,
        "chinese-xlnet-large": None,
    }
    max_model_input_sizes = pretrained_positional_embedding_sizes
    padding_side = "left"
    pad_token_type_id = 3

    def __init__(
            self,
            vocab_file,
            do_lower_case=False,
            remove_space=True,
            keep_accents=False,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
            additional_special_tokens=["<eop>", "<eod>"], ):

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file
        spm = try_import("sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def get_vocab(self):
        vocab = {
            self.convert_ids_to_tokens(i): i
            for i in range(self.vocab_size)
        }
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        spm = try_import("sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join(
                [c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text, sample=False):
        """Tokenize a string."""
        text = self.preprocess_text(text)

        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(
                    SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][
                        0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def _convert_token_to_id(self, token):
        """Converts a token (str) to an id using the vocab. """
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) to a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        # Converts a sequence of tokens (strings for sub-words) in a single string.
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Args:
            pair (bool, optional):
                Whether the input is a sequence pair or a single sequence.
                Defaults to `False` and the input is a single sequence.

        Returns:
            int: Number of tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Builds model inputs from a sequence or a pair of sequence
        for sequence classification tasks by concatenating and
        adding special tokens. An XLNet sequence has the following format:

        - single sequence:      ``X <sep> <cls>``
        - pair of sequences:    ``A <sep> B <sep> <cls>``

        Args:
            token_ids_0 (List[int]):
                List of IDs for the first sequence.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for the second sequenze. Defaults to `None`.

        Returns:
            List[int]: List of input IDs with the appropriate special tokens.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return token_ids_0 + sep + cls
        return token_ids_0 + sep + token_ids_1 + sep + cls

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        """
        Builds offset map from a pair of offset map by concatenating
        and adding offsets of special tokens.

        An XLNet offset_mapping has the following format:

        - single sequence:      ``X (0,0) (0,0)``
        - pair of sequences:    ``A (0,0) B (0,0) (0,0)``

        Args:
            offset_mapping_0 (List[tuple]):
                List of char offsets to which the special tokens will be added.
            offset_mapping_1 (List[tuple], optional):
                Optional second list of char offsets for offset mapping pairs.
                Defaults to `None`.

        Returns:
            List[tuple]: A list of char offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return offset_mapping_0 + [(0, 0)] + [(0, 0)]

        return offset_mapping_0 + [(0, 0)] + offset_mapping_1 + [(0, 0)] + [
            (0, 0)
        ]

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """
        Creates a special tokens mask from the input sequences.
        This method is called when adding special tokens using the tokenizer `encode` method.

        Args:
            token_ids_0 (List[int]):
                A list of `inputs_ids` for the first sequence.
            token_ids_1 (List[int], optional):
                Optional second list of `inputs_ids` for the second sequence.
                Defaults to `None`.
            already_has_special_tokens (bool, optional):
                Whether or not the token list already contains special tokens for the model.
                Defaults to `False`.

        Returns:
            List[int]: A list of integers which is either 0 or 1: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0))

        if token_ids_1 is not None:
            return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)
                                                     ) + [1, 1]
        return ([0] * len(token_ids_0)) + [1, 1]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Creates a token_type mask from the input sequences.
        If `token_ids_1` is not `None`, then a sequence pair
        token_type mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 2
            | first sequence    | second sequence |

        Else if `token_ids_1` is `None`, then a single sequence
        token_type mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
            |            first sequence           |

        - 0 stands for the segment id of **first segment tokens**,
        - 1 stands for the segment id of **second segment tokens**,
        - 2 stands for the segment id of **cls_token**.

        Args:
            token_ids_0 (List[int]):
                A list of `inputs_ids` for the first sequence.
            token_ids_1 (List[int], optional):
                Optional second list of `inputs_ids` for the second sequence.
                Defaults to `None`.

        Returns:
            List[int]: List of token type IDs according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls_segment_id = [2]

        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0] + cls_segment_id
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 +
                                                  sep) * [1] + cls_segment_id

    def save_resources(self, save_directory):
        """
        Saves `SentencePiece <https://github.com/google/sentencepiece>`__ file
        (ends with '.spm') under `save_directory`.

        Args:
            save_directory (str):
                Directory to save files into.
        """
        for name, file_name in self.resource_files_names.items():
            save_path = os.path.join(save_directory, file_name)
            if os.path.abspath(self.vocab_file) != os.path.abspath(save_path):
                copyfile(self.vocab_file, save_path)
