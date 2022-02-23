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
""" Tokenization classes for LayoutLM model."""
from ..bert.tokenizer import BertTokenizer

__all__ = ['LayoutLMTokenizer']


class LayoutLMTokenizer(BertTokenizer):
    """
    The usage of LayoutLMTokenizer is the same as
    `BertTokenizer <https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.bert.tokenizer.html>`__.
    For more information regarding those methods, please refer to this superclass.
    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "layoutlm-base-uncased":
            "https://bj.bcebos.com/paddlenlp/models/transformers/layoutlm/layoutlm-base-uncased/vocab.txt",
            "layoutlm-large-uncased":
            "https://bj.bcebos.com/paddlenlp/models/transformers/layoutlm/layoutlm-large-uncased/vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "layoutlm-base-uncased": {
            "do_lower_case": True
        },
        "layoutlm-large-uncased": {
            "do_lower_case": True
        },
    }
