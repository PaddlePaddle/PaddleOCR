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

from ..bert.tokenizer import BertTokenizer

__all__ = ['TinyBertTokenizer']


class TinyBertTokenizer(BertTokenizer):
    """
    Constructs a TinyBert tokenizer. The usage of TinyBertTokenizer is the same as
    `BertTokenizer <https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.bert.tokenizer.html>`__.
    For more information regarding those methods, please refer to this superclass.
    """

    pretrained_resource_files_map = {
        "vocab_file": {
            "tinybert-4l-312d":
            "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-4l-312d-vocab.txt",
            "tinybert-6l-768d":
            "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-6l-768d-vocab.txt",
            "tinybert-4l-312d-v2":
            "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-4l-312d-v2-vocab.txt",
            "tinybert-6l-768d-v2":
            "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-6l-768d-v2-vocab.txt",
            "tinybert-4l-312d-zh":
            "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-4l-312d-zh-vocab.txt",
            "tinybert-6l-768d-zh":
            "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-6l-768d-zh-vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "tinybert-4l-312d": {
            "do_lower_case": True
        },
        "tinybert-6l-768d": {
            "do_lower_case": True
        },
        "tinybert-4l-312d-v2": {
            "do_lower_case": True
        },
        "tinybert-6l-768d-v2": {
            "do_lower_case": True
        },
        "tinybert-4l-312d-zh": {
            "do_lower_case": True
        },
        "tinybert-6l-768d-zh": {
            "do_lower_case": True
        },
    }
