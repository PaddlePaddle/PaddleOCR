# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import paddle
import paddle.nn as nn


class ErnieForCSC(nn.Layer):
    r"""
    ErnieForCSC is a model specified for Chinese Spelling Correction task.

    It integrates phonetic features into language model by leveraging the powerful
    pre-training and fine-tuning method.

    See more details on https://aclanthology.org/2021.findings-acl.198.pdf.
    Args:
        ernie (ErnieModel): 
            An instance of `paddlenlp.transformers.ErnieModel`.
        pinyin_vocab_size (int): 
            The vocab size of pinyin vocab.
        pad_pinyin_id (int, optional): 
            The pad token id of pinyin vocab. Defaults to 0.
    """

    def __init__(self, ernie, pinyin_vocab_size, pad_pinyin_id=0):
        super(ErnieForCSC, self).__init__()
        self.ernie = ernie
        emb_size = self.ernie.config["hidden_size"]
        hidden_size = self.ernie.config["hidden_size"]
        vocab_size = self.ernie.config["vocab_size"]

        self.pad_token_id = self.ernie.config["pad_token_id"]
        self.pinyin_vocab_size = pinyin_vocab_size
        self.pad_pinyin_id = pad_pinyin_id
        self.pinyin_embeddings = nn.Embedding(
            self.pinyin_vocab_size, emb_size, padding_idx=pad_pinyin_id)
        self.detection_layer = nn.Linear(hidden_size, 2)
        self.correction_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax()

    def forward(self,
                input_ids,
                pinyin_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            pinyin_ids (Tensor):
                Indices of pinyin tokens of input sequence in the pinyin vocabulary. They are
                numerical representations of tokens that build the pinyin input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                Segment token indices to indicate first and second portions of the inputs.
                Indices can be either 0 or 1:

                - 0 corresponds to a **sentence A** token,
                - 1 corresponds to a **sentence B** token.

                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
                Defaults to None, which means no segment embeddings is added to token embeddings.
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                config.max_position_embeddings - 1]``.
                Defaults to `None`. Shape as `(batch_sie, num_tokens)` and dtype as `int32` or `int64`.
            attention_mask (Tensor, optional):
                Mask to indicate whether to perform attention on each input token or not.
                The values should be either 0 or 1. The attention scores will be set
                to **-infinity** for any positions in the mask that are **0**, and will be
                **unchanged** for positions that are **1**.

                - **1** for tokens that are **not masked**,
                - **0** for tokens that are **masked**.

                It's data type should be `float32` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.


        Returns:
            det_preds (Tensor):
                A Tensor of the detection prediction of each tokens.
                Shape as `(batch_size, sequence_length)` and dtype as `int`.

            char_preds (Tensor):
                A Tensor of the correction prediction of each tokens.
                Shape as `(batch_size, sequence_length)` and dtype as `int`.

        """
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.detection_layer.weight.dtype) * -1e4,
                axis=[1, 2])

        embedding_output = self.ernie.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        pinyin_embedding_output = self.pinyin_embeddings(pinyin_ids)

        # Detection module aims to detect whether each Chinese charater has spelling error.
        detection_outputs = self.ernie.encoder(embedding_output, attention_mask)
        # detection_error_probs shape: [B, T, 2]. It indicates the erroneous probablity of each 
        # word in the sequence from 0 to 1.
        detection_error_probs = self.softmax(
            self.detection_layer(detection_outputs))
        # Correction module aims to correct each potential wrong charater to right charater.
        word_pinyin_embedding_output = detection_error_probs[:, :, 0:1] * embedding_output \
                    + detection_error_probs[:,:, 1:2] * pinyin_embedding_output

        correction_outputs = self.ernie.encoder(word_pinyin_embedding_output,
                                                attention_mask)
        # correction_logits shape: [B, T, V]. It indicates the correct score of each token in vocab 
        # according to each word in the sequence.
        correction_logits = self.correction_layer(correction_outputs)

        det_preds = detection_error_probs.argmax(axis=-1)
        char_preds = correction_logits.argmax(axis=-1)
        return det_preds, char_preds
