# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The HuggingFace Inc. team, Microsoft Corporation.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from ..bert.tokenizer import BertTokenizer
from .. import AddedToken

__all__ = ['MPNetTokenizer']


class MPNetTokenizer(BertTokenizer):
    """
    Construct a MPNet tokenizer which is almost identical to `BertTokenizer`.
    For more information regarding those methods, please refer to this superclass.
    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "mpnet-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/mpnet/mpnet-base/vocab.txt",
        }
    }
    pretrained_init_configuration = {"mpnet-base": {"do_lower_case": True}}

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 bos_token="<s>",
                 eos_token="</s>",
                 unk_token="[UNK]",
                 sep_token="</s>",
                 pad_token="<pad>",
                 cls_token="<s>",
                 mask_token="<mask>",
                 **kwargs):

        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token)

        bos_token = AddedToken(
            bos_token, lstrip=False,
            rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(
            eos_token, lstrip=False,
            rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(
            sep_token, lstrip=False,
            rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(
            cls_token, lstrip=False,
            rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(
            unk_token, lstrip=False,
            rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(
            pad_token, lstrip=False,
            rstrip=False) if isinstance(pad_token, str) else pad_token

        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(
            mask_token, lstrip=True,
            rstrip=False) if isinstance(mask_token, str) else mask_token
        self._build_special_tokens_map_extended(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token)

    def __call__(self,
                 text,
                 text_pair=None,
                 max_seq_len=None,
                 stride=0,
                 is_split_into_words=False,
                 pad_to_max_seq_len=False,
                 truncation_strategy="longest_first",
                 return_position_ids=False,
                 return_token_type_ids=False,
                 return_attention_mask=False,
                 return_length=False,
                 return_overflowing_tokens=False,
                 return_special_tokens_mask=False):
        return super().__call__(
            text,
            text_pair=text_pair,
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

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. 
        
        A MPNet sequence has the following format:

        - single sequence:      ``<s> X </s>``
        - pair of sequences:        ``<s> A </s></s> B </s>``

        Args:
            token_ids_0 (List[int]):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs. Defaults to None.

        Returns:
            List[int]: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``encode`` methods.

        Args:
            token_ids_0 (List[int]):
                A list of `inputs_ids` for the first sequence.
            token_ids_1 (List[int], optinal):
                Optional second list of IDs for sequence pairs. Defaults to None.
            already_has_special_tokens (bool, optional): Whether or not the token list is already 
                formatted with special tokens for the model. Defaults to None.

        Returns:
            List[int]: The list of integers either be 0 or 1: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True)

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)
                                                          ) + [1]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. MPNet does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (List[int]):
                A list of `inputs_ids` for the first sequence.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs. Defaults to None.

        Returns:
            List[int]: List of token_type_id according to the given sequence(s).
        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
