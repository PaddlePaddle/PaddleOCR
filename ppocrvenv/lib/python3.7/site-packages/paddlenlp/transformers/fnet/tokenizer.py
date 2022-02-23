# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization class for FNet model."""

import os
import unicodedata
import sentencepiece as spm
from shutil import copyfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from paddlenlp.transformers.albert.tokenizer import AlbertEnglishTokenizer

__all__ = ['FNetTokenizer']

SPIECE_UNDERLINE = "▁"


@dataclass(frozen=True, eq=True)
class AddedToken:
    """
    AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
    way it should behave.
    Copied from transformers.tokenization_utils_base
    """
    content: str = field(default_factory=str)
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = True
    
    def __getstate__(self):
        return self.__dict__


class FNetTokenizer(AlbertEnglishTokenizer):
    """
        Construct a FNet tokenizer. Inherit from :class:`AlbertEnglishTokenizer`. Based on `SentencePiece
        <https://github.com/google/sentencepiece>`__.

        Args:
            sentencepiece_model_file (:obj:`str`):
                `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
                contains the vocabulary necessary to instantiate a tokenizer.
            do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to lowercase the input when tokenizing.
            remove_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).
            keep_accents (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to keep accents when tokenizing.
            unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
                token instead.
            sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
                The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
                sequence classification or for a text and a question for question answering. It is also used as the last
                token of a sequence built with special tokens.
            pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
                The token used for padding, for example when batching sequences of different lengths.
            cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
                The classifier token which is used when doing sequence classification (classification of the whole sequence
                instead of per-token classification). It is the first token of the sequence when built with special tokens.
            mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
                The token used for masking values. This is the token used when training this model with masked language
                modeling. This is the token which the model will try to predict.
            sp_model_kwargs (:obj:`dict`, `optional`):
                Will be passed to the ``SentencePieceProcessor.__init__()`` method. The `Python wrapper for SentencePiece
                <https://github.com/google/sentencepiece/tree/master/python>`__ can be used, among other things, to set:

                - ``enable_sampling``: Enable subword regularization.
                - ``nbest_size``: Sampling parameters for unigram. Invalid for BPE-Dropout.

                  - ``nbest_size = {0,1}``: No sampling is performed.
                  - ``nbest_size > 1``: samples from the nbest_size results.
                  - ``nbest_size < 0``: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                    using forward-filtering-and-backward-sampling algorithm.
                - ``alpha``: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
                  BPE-dropout.

        Attributes:
            sp_model (:obj:`SentencePieceProcessor`):
                The `SentencePiece` processor that is used for every conversion (string, tokens and IDs).
        """
    resource_files_names = {
        "sentencepiece_model_file": "spiece.model",
    }
    pretrained_resource_files_map = {
        "sentencepiece_model_file": {
            "fnet-base": "https://bj.bcebos.com/paddlenlp/models/transformers/fnet/fnet-base/spiece.model",
            "fnet-large": "https://bj.bcebos.com/paddlenlp/models/transformers/fnet/fnet-large/spiece.model",
        }
    }
    pretrained_init_configuration = {
        "fnet-base": {
            "do_lower_case": False,
        },
        "fnet-large": {
            "do_lower_case": False,
        }
    }
    model_input_names = ["input_ids", "token_type_ids"]
    
    def __init__(self, sentencepiece_model_file, do_lower_case=False, remove_space=True, keep_accents=True,
                 unk_token="<unk>", sep_token="[SEP]", pad_token="<pad>", cls_token="[CLS]", mask_token="[MASK]",
                 sp_model_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        # Mask token behave like a normal word, i.e. include the space before it
        # mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        
        super().__init__(sentencepiece_model_file, do_lower_case, remove_space, keep_accents, unk_token, sep_token,
                         pad_token, cls_token, mask_token, **kwargs)
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sentencepiece_model_file)
    
    @property
    def vocab_size(self):
        return len(self.sp_model)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state
    
    def __setstate__(self, d):
        self.__dict__ = d
        
        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}
        
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
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()
        
        return outputs
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string."""
        text = self.preprocess_text(text)
        pieces = self.sp_model.EncodeAsPieces(text)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(
                    SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)
        
        return new_pieces
    
    def tokenize(self, text):
        return self._tokenize(text)
    
    def _convert_token_to_id(self, token):
        """Converts a token (str) to an id using the vocab. """
        return self.sp_model.PieceToId(token)
    
    def _convert_id_to_token(self, index):
        """Converts an index (integer) to a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)
    
    def convert_tokens_to_ids(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._convert_token_to_id(tokens)
        else:
            return [self._convert_token_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if not isinstance(ids, (list, tuple)):
            return self._convert_id_to_token(ids)
        tokens = [self._convert_id_to_token(_id) for _id in ids]
        return tokens
    
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string
    
    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An FNet sequence has the following format:

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
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep
    
    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        
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
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                    [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
    
    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An FNet sequence
        pair mask has the following format: ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 | first sequence | second sequence |

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
    
    def save_resources(self, save_directory):
        for name, file_name in self.resource_files_names.items():
            save_path = os.path.join(save_directory, file_name)
            if os.path.abspath(self.sentencepiece_model_file
                               ) != os.path.abspath(save_path):
                copyfile(self.sentencepiece_model_file, save_path)
