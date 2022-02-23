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

__all__ = ['DistilBertTokenizer']


class DistilBertTokenizer(BertTokenizer):
    """
    Constructs a DistilBert tokenizer. The usage of DistilBertTokenizer is the same as
    `BertTokenizer <https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.bert.tokenizer.html>`__.
    For more information regarding those methods, please refer to this superclass.
    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "distilbert-base-uncased":
            "https://bj.bcebos.com/paddlenlp/models/transformers/distilbert/distilbert-base-uncased-vocab.txt",
            "distilbert-base-cased":
            "https://bj.bcebos.com/paddlenlp/models/transformers/distilbert/distilbert-base-cased-vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "distilbert-base-uncased": {
            "do_lower_case": True
        },
        "distilbert-base-cased": {
            "do_lower_case": False
        },
    }

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
        return super(DistilBertTokenizer, self).__call__(
            text, text_pair, max_seq_len, stride, is_split_into_words,
            pad_to_max_seq_len, truncation_strategy, return_position_ids,
            return_token_type_ids, return_attention_mask, return_length,
            return_overflowing_tokens, return_special_tokens_mask)
