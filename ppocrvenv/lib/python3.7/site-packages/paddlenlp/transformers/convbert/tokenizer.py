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

from ..electra.tokenizer import ElectraTokenizer

__all__ = ['ConvBertTokenizer', ]


class ConvBertTokenizer(ElectraTokenizer):
    """
    Construct a ConvBERT tokenizer. `ConvBertTokenizer` is identical to `ElectraTokenizer`.
    For more information regarding those methods, please refer to this superclass.
    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "convbert-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/convbert/convbert-base/vocab.txt",
            "convbert-medium-small":
            "https://bj.bcebos.com/paddlenlp/models/transformers/convbert/convbert-medium-small/vocab.txt",
            "convbert-small":
            "https://bj.bcebos.com/paddlenlp/models/transformers/convbert/convbert-small/vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "convbert-base": {
            "do_lower_case": True
        },
        "convbert-medium-small": {
            "do_lower_case": True
        },
        "convbert-small": {
            "do_lower_case": True
        },
    }
