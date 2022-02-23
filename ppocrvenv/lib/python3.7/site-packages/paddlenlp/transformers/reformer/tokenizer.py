# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.utils import try_import
from ..albert.tokenizer import AlbertEnglishTokenizer

__all__ = ['ReformerTokenizer']


class ReformerTokenizer(AlbertEnglishTokenizer):
    """
    Constructs a Reformer tokenizer based on SentencePiece .
    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.

    Args:
        sentencepiece_model_file (str):
            The vocabulary file (ends with '.spm') required to instantiate
            a `SentencePiece <https://github.com/google/sentencepiece>`__ tokenizer.
        do_lower_case (bool):
            Whether or not to lowercase the input when tokenizing. Defaults to `False`.
        remove_space (bool):
            Whether or note to remove space when tokenizing. Defaults to `True`.
        keep_accents (bool):
            Whether or note to keep accents when tokenizing. Defaults to `False`.
        eos_token (str):
            A special token representing the *eos (end-of-sentence)* token.
            Defaults to "</s>".
        unk_token (str):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to "<unk>".
        pad_token (str):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to "<unk>".

    """

    resource_files_names = {"sentencepiece_model_file": "spiece.model", }
    pretrained_resource_files_map = {
        "sentencepiece_model_file": {
            "reformer-crime-and-punishment":
            "http://paddlenlp.bj.bcebos.com/models/transformers/reformer/reformer-crime-and-punishment/spiece.model",
        },
    }

    pretrained_init_configuration = {
        "reformer-crime-and-punishment": {
            "do_lower_case": False
        },
    }

    def __init__(self,
                 sentencepiece_model_file,
                 do_lower_case=False,
                 remove_space=True,
                 keep_accents=False,
                 eos_token="</s>",
                 unk_token="<unk>",
                 pad_token="<unk>",
                 **kwargs):

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.sentencepiece_model_file = sentencepiece_model_file

        spm = try_import("sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sentencepiece_model_file)

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
                 return_attention_mask=True,
                 return_length=False,
                 return_overflowing_tokens=False,
                 return_special_tokens_mask=False):
        return super(ReformerTokenizer, self).__call__(
            text, text_pair, max_seq_len, stride, is_split_into_words,
            pad_to_max_seq_len, truncation_strategy, return_position_ids,
            return_token_type_ids, return_attention_mask, return_length,
            return_overflowing_tokens, return_special_tokens_mask)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence.

        An Reformer sequence has the following format:

        - single sequence:      ``X``
        - pair of sequences:        ``A B ``

        Args:
            token_ids_0 (List[int]):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs. Defaults to None.

        Returns:
            List[int]: List of input_id with the appropriate special tokens.

        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences.

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
        return len(token_ids_0) * [0] + len(token_ids_1) * [1]
