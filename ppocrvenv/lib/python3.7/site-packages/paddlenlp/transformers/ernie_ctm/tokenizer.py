# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.

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

import os
import pickle
import six
import shutil

from paddle.utils import try_import
from paddlenlp.utils.env import MODEL_HOME

from .. import BasicTokenizer, PretrainedTokenizer

__all__ = ['ErnieCtmTokenizer']


class ErnieCtmTokenizer(PretrainedTokenizer):
    r"""
    Construct an ERNIE-CTM tokenizer.

    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.
    
    Args:
        vocab_file (str):
            File path of the vocabulary.
        do_lower_case (bool, optional):
            Whether or not to lowercase the input when tokenizing. Defaults to `True`
        do_basic_tokenize (bool, optional):
            Whether or not to do basic tokenization before WordPiece. Defaults to `True`
        unk_token (str, optional):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to "[UNK]".
        sep_token (str, optional):
            A special token separating two different sentences in the same input.
            Defaults to "[SEP]".
        pad_token (str, optional):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to "[PAD]".
        cls_token_template (str, optional)
            The template of summary token for multiple summary placeholders. Defaults to `"[CLS{}]"`
        cls_num (int, optional):
            Summary placeholder used in ernie-ctm model. For catching a sentence global feature from multiple aware.
            Defaults to `1`.
        mask_token (str, optional):
            A special token representing a masked token. This is the token used in the masked
            language modeling task. This is the token which the model will try to predict the original unmasked ones.
            Defaults to `"[MASK]"`.
        strip_accents: (bool, optional):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).

    Examples:
        .. code-block::

            from paddlenlp.transformers import ErnieCtmTokenizer
            tokenizer = ErnieCtmTokenizer.from_pretrained('ernie-ctm')

            encoded_inputs = tokenizer('He was a puppeteer')
            # encoded_inputs:
            # {'input_ids': [101, 98, 153, 150, 99, 168, 146, 164, 99, 146, 99, 161, 166, 161,
            #  161, 150, 165, 150, 150, 163, 102],
            # 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-ctm":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_ctm/vocab.txt",
            "wordtag":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_ctm/vocab.txt",
            "nptag":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_ctm/vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "ernie-ctm": {
            "do_lower_case": True,
            "cls_num": 2
        },
        "wordtag": {
            "do_lower_case": True,
            "cls_num": 2
        },
        "nptag": {
            "do_lower_case": True,
            "cls_num": 2
        },
    }

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 do_basic_tokenize=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token_template="[CLS{}]",
                 cls_num=1,
                 mask_token="[MASK]",
                 **kwargs):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = ErnieTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.do_lower_case = do_lower_case
        self.cls_token_template = cls_token_template
        self.cls_num = cls_num
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)

    @property
    def vocab_size(self):
        """
        Return the size of vocabulary.

        Returns:
            int: The size of vocabulary.
        """
        return len(self.vocab)

    def convert_tokens_to_string(self, tokens):
        r"""
        Converts a sequence of tokens (list of string) in a single string. Since
        the usage of WordPiece introducing `##` to concat subwords, also remove
        `##` when converting.

        Args:
            tokens (List[str]): A list of string representing tokens to be converted.

        Returns:
            str: Converted string from tokens.

        Examples:
            .. code-block::

                from paddlenlp.transformers import ErnieCtmTokenizer
                tokenizer = ErnieCtmTokenizer.from_pretrained('ernie-ctm')

                tokens = tokenizer.tokenize('He was a puppeteer')
                strings = tokenizer.convert_tokens_to_string(tokens)
                #he was a puppeteer

        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by
        concatenating and add special tokens.

        A ERNIE-CTM sequence has the following format:

        - single sequence:      [CLS0][CLS1]... X [SEP]
        - pair of sequences:        [CLS0][CLS1]... X [SEP] X [SEP]

        Args:
            token_ids_0 (List):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List, optional):
                Optional second list of IDs for sequence pairs. Defaults to ``None``.

        Returns:
            List[int]: The input_id with the appropriate special tokens.
        """
        cls_token_ids = [
            self.convert_tokens_to_ids(self.cls_token_template.format(sid))
            for sid in range(self.cls_num)
        ]
        if token_ids_1 is None:
            return cls_token_ids + token_ids_0 + [self.sep_token_id]
        return cls_token_ids + token_ids_0 + [
            self.sep_token_id
        ] + token_ids_1 + [self.sep_token_id]

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
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

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
        if token_ids_1 is None:
            return (self.cls_num + len(token_ids_0 + sep)) * [0]
        return (self.cls_num + len(token_ids_0 + sep)
                ) * [0] + len(token_ids_1 + sep) * [1]

    def num_special_tokens_to_add(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient.
            Do not put this inside your training loop.

        Args:
            pair (bool, optional):
                Whether the input is a sequence pair or a single sequence.
                Defaults to `False` and the input is a single sequence.

        Returns:
            int: Number of tokens added to sequences.
        """
        if pair is True:
            return self.cls_num + 2
        else:
            return self.cls_num + 1

    def _tokenize(self, text, **kwargs):
        r"""
        Converts a string to a list of tokens.

        Args:
            text (str): The text to be tokenized.
        
        Returns:
            List[str]: A list of string representing converted tokens.
        """
        orig_tokens = list(text)
        output_tokens = []
        for token in orig_tokens:
            if self.do_lower_case is True:
                token = token.lower()
            output_tokens.append(token)
        return output_tokens
