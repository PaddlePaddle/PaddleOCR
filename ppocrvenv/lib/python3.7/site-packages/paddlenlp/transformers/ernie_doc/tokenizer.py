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
import os
import pickle
import shutil
import json

from paddlenlp.utils.env import MODEL_HOME
from .. import PretrainedTokenizer, BPETokenizer
from ..ernie.tokenizer import ErnieTokenizer

__all__ = ['ErnieDocTokenizer', 'ErnieDocBPETokenizer']


class ErnieDocTokenizer(ErnieTokenizer):
    r"""
    Constructs an ERNIE-Doc tokenizer.
    It uses a basic tokenizer to do punctuation splitting, lower casing and so on,
    and follows a WordPiece tokenizer to tokenize as subwords.

    This tokenizer inherits from :class:`~paddlenlp.transformers.ernie.tokenizer.ErnieTokenizer`.
    For more information regarding those methods, please refer to this superclass.

    Args:
        vocab_file (str):
            The vocabulary file path (ends with '.txt') required to instantiate
            a `WordpieceTokenizer`.
        do_lower_case (str, optional):
            Whether or not to lowercase the input when tokenizing.
            Defaults to`True`.
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
        cls_token (str, optional):
            A special token used for sequence classification. It is the last token
            of the sequence when built with special tokens. Defaults to "[CLS]".
        mask_token (str, optional):
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to "[MASK]".
    
    Examples:
        .. code-block::

            from paddlenlp.transformers import ErnieDocTokenizer
            tokenizer = ErnieDocTokenizer.from_pretrained('ernie-doc-base-zh')
            encoded_inputs = tokenizer('He was a puppeteer')

    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-doc-base-zh":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-doc-base-zh/vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "ernie-doc-base-zh": {
            "do_lower_case": True
        },
    }

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 **kwargs):
        super(ErnieDocTokenizer, self).__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token)


class ErnieDocBPETokenizer(BPETokenizer):
    r"""
    Constructs an ERNIE-Doc BPE tokenizer. It uses a bpe tokenizer to do punctuation
    splitting, lower casing and so on, then tokenize words as subwords.

    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.BPETokenizer`.
    For more information regarding those methods, please refer to this superclass.

    Args:
        vocab_file (str): 
            File path of the vocabulary.
        encoder_json_path (str, optional):
            File path of the id to vocab.
        vocab_bpe_path (str, optional):
            File path of word merge text.
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
        cls_token (str, optional):
            A special token used for sequence classification. It is the last token
            of the sequence when built with special tokens. Defaults to "[CLS]".
        mask_token (str, optional):
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to "[MASK]".
    
    Examples:
        .. code-block::

            from paddlenlp.transformers import ErnieDocBPETokenizer
            tokenizer = ErnieDocBPETokenizer.from_pretrained('ernie-doc-base-en')
            encoded_inputs = tokenizer('He was a puppeteer')

    """
    resource_files_names = {
        "vocab_file": "vocab.txt",
        "encoder_json_path": "encoder.json",
        "vocab_bpe_path": "vocab.bpe"
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-doc-base-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-doc-base-en/vocab.txt"
        },
        "encoder_json_path": {
            "ernie-doc-base-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-doc-base-en/encoder.json"
        },
        "vocab_bpe_path": {
            "ernie-doc-base-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-doc-base-en/vocab.bpe"
        }
    }
    pretrained_init_configuration = {
        "ernie-doc-base-en": {
            "unk_token": "[UNK]"
        },
    }

    def __init__(self,
                 vocab_file,
                 encoder_json_path="./configs/encoder.json",
                 vocab_bpe_path="./configs/vocab.bpe",
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 **kwargs):
        super(ErnieDocBPETokenizer, self).__init__(
            vocab_file,
            encoder_json_path=encoder_json_path,
            vocab_bpe_path=vocab_bpe_path,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token)

    @property
    def vocab_size(self):
        """
        Return the size of vocabulary.

        Returns:
            int: The size of vocabulary.
        """
        return len(self.vocab)
