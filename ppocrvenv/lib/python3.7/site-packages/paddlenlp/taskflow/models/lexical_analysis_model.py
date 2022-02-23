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
import paddle.nn.functional as F
from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss
from paddlenlp.utils.tools import compare_version

if compare_version(paddle.version.full_version, "2.2.0") >= 0:
    # paddle.text.ViterbiDecoder is supported by paddle after version 2.2.0
    from paddle.text import ViterbiDecoder
else:
    from paddlenlp.layers.crf import ViterbiDecoder


class BiGruCrf(nn.Layer):
    """The network for lexical analysis, based on two layers of BiGRU and one layer of CRF. More details see https://arxiv.org/abs/1807.01882
    Args:
        word_emb_dim (int): The dimension in which a word is embedded.
        hidden_size (int): The number of hidden nodes in the GRU layer.
        vocab_size (int): the word vocab size.
        num_labels (int): the labels amount.
        emb_lr (float, optional): The scaling of the learning rate of the embedding layer. Defaults to 2.0.
        crf_lr (float, optional): The scaling of the learning rate of the crf layer. Defaults to 0.2.
    """

    def __init__(self,
                 word_emb_dim,
                 hidden_size,
                 vocab_size,
                 num_labels,
                 emb_lr=2.0,
                 crf_lr=0.2,
                 with_start_stop_tag=True):
        super(BiGruCrf, self).__init__()
        self.word_emb_dim = word_emb_dim
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.emb_lr = emb_lr
        self.crf_lr = crf_lr
        self.init_bound = 0.1

        self.word_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.word_emb_dim,
            weight_attr=paddle.ParamAttr(
                learning_rate=self.emb_lr,
                initializer=nn.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound)))

        self.gru = nn.GRU(
            input_size=self.word_emb_dim,
            hidden_size=self.hidden_size,
            num_layers=2,
            direction='bidirectional',
            weight_ih_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound),
                regularizer=paddle.regularizer.L2Decay(coeff=1e-4)),
            weight_hh_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound),
                regularizer=paddle.regularizer.L2Decay(coeff=1e-4)))

        self.fc = nn.Linear(
            in_features=self.hidden_size * 2,
            out_features=self.num_labels + 2 \
                if with_start_stop_tag else self.num_labels,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound),
                regularizer=paddle.regularizer.L2Decay(coeff=1e-4)))

        self.crf = LinearChainCrf(self.num_labels, self.crf_lr,
                                  with_start_stop_tag)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions,
                                              with_start_stop_tag)

    def forward(self, inputs, lengths, labels=None):
        word_embed = self.word_embedding(inputs)
        bigru_output, _ = self.gru(word_embed, sequence_length=lengths)
        emission = self.fc(bigru_output)
        if labels is not None:
            loss = self.crf_loss(emission, lengths, labels)
            return loss
        else:
            _, prediction = self.viterbi_decoder(emission, lengths)
            return prediction
