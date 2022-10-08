# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn
from paddle import ParamAttr
import paddle.nn.functional as F
import numpy as np

from .rec_att_head import AttentionGRUCell


def get_para_bias_attr(l2_decay, k):
    if l2_decay > 0:
        regularizer = paddle.regularizer.L2Decay(l2_decay)
        stdv = 1.0 / math.sqrt(k * 1.0)
        initializer = nn.initializer.Uniform(-stdv, stdv)
    else:
        regularizer = None
        initializer = None
    weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    return [weight_attr, bias_attr]


class TableAttentionHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 in_max_len=488,
                 max_text_length=800,
                 out_channels=30,
                 loc_reg_num=4,
                 **kwargs):
        super(TableAttentionHead, self).__init__()
        self.input_size = in_channels[-1]
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.max_text_length = max_text_length

        self.structure_attention_cell = AttentionGRUCell(
            self.input_size, hidden_size, self.out_channels, use_gru=False)
        self.structure_generator = nn.Linear(hidden_size, self.out_channels)
        self.in_max_len = in_max_len

        if self.in_max_len == 640:
            self.loc_fea_trans = nn.Linear(400, self.max_text_length + 1)
        elif self.in_max_len == 800:
            self.loc_fea_trans = nn.Linear(625, self.max_text_length + 1)
        else:
            self.loc_fea_trans = nn.Linear(256, self.max_text_length + 1)
        self.loc_generator = nn.Linear(self.input_size + hidden_size,
                                       loc_reg_num)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None):
        # if and else branch are both needed when you want to assign a variable
        # if you modify the var in just one branch, then the modification will not work.
        fea = inputs[-1]
        last_shape = int(np.prod(fea.shape[2:]))  # gry added
        fea = paddle.reshape(fea, [fea.shape[0], fea.shape[1], last_shape])
        fea = fea.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
        batch_size = fea.shape[0]

        hidden = paddle.zeros((batch_size, self.hidden_size))
        output_hiddens = paddle.zeros(
            (batch_size, self.max_text_length + 1, self.hidden_size))
        if self.training and targets is not None:
            structure = targets[0]
            for i in range(self.max_text_length + 1):
                elem_onehots = self._char_to_onehot(
                    structure[:, i], onehot_dim=self.out_channels)
                (outputs, hidden), alpha = self.structure_attention_cell(
                    hidden, fea, elem_onehots)
                output_hiddens[:, i, :] = outputs
            structure_probs = self.structure_generator(output_hiddens)
            loc_fea = fea.transpose([0, 2, 1])
            loc_fea = self.loc_fea_trans(loc_fea)
            loc_fea = loc_fea.transpose([0, 2, 1])
            loc_concat = paddle.concat([output_hiddens, loc_fea], axis=2)
            loc_preds = self.loc_generator(loc_concat)
            loc_preds = F.sigmoid(loc_preds)
        else:
            temp_elem = paddle.zeros(shape=[batch_size], dtype="int32")
            structure_probs = None
            loc_preds = None
            elem_onehots = None
            outputs = None
            alpha = None
            max_text_length = paddle.to_tensor(self.max_text_length)
            for i in range(max_text_length + 1):
                elem_onehots = self._char_to_onehot(
                    temp_elem, onehot_dim=self.out_channels)
                (outputs, hidden), alpha = self.structure_attention_cell(
                    hidden, fea, elem_onehots)
                output_hiddens[:, i, :] = outputs
                structure_probs_step = self.structure_generator(outputs)
                temp_elem = structure_probs_step.argmax(axis=1, dtype="int32")

            structure_probs = self.structure_generator(output_hiddens)
            structure_probs = F.softmax(structure_probs)
            loc_fea = fea.transpose([0, 2, 1])
            loc_fea = self.loc_fea_trans(loc_fea)
            loc_fea = loc_fea.transpose([0, 2, 1])
            loc_concat = paddle.concat([output_hiddens, loc_fea], axis=2)
            loc_preds = self.loc_generator(loc_concat)
            loc_preds = F.sigmoid(loc_preds)
        return {'structure_probs': structure_probs, 'loc_preds': loc_preds}


class SLAHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 out_channels=30,
                 max_text_length=500,
                 loc_reg_num=4,
                 fc_decay=0.0,
                 **kwargs):
        """
        @param in_channels: input shape
        @param hidden_size: hidden_size for RNN and Embedding
        @param out_channels: num_classes to rec
        @param max_text_length: max text pred
        """
        super().__init__()
        in_channels = in_channels[-1]
        self.hidden_size = hidden_size
        self.max_text_length = max_text_length
        self.emb = self._char_to_onehot
        self.num_embeddings = out_channels
        self.loc_reg_num = loc_reg_num

        # structure
        self.structure_attention_cell = AttentionGRUCell(
            in_channels, hidden_size, self.num_embeddings)
        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=fc_decay, k=hidden_size)
        weight_attr1_1, bias_attr1_1 = get_para_bias_attr(
            l2_decay=fc_decay, k=hidden_size)
        weight_attr1_2, bias_attr1_2 = get_para_bias_attr(
            l2_decay=fc_decay, k=hidden_size)
        self.structure_generator = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                self.hidden_size,
                weight_attr=weight_attr1_2,
                bias_attr=bias_attr1_2),
            nn.Linear(
                hidden_size,
                out_channels,
                weight_attr=weight_attr,
                bias_attr=bias_attr))
        # loc
        weight_attr1, bias_attr1 = get_para_bias_attr(
            l2_decay=fc_decay, k=self.hidden_size)
        weight_attr2, bias_attr2 = get_para_bias_attr(
            l2_decay=fc_decay, k=self.hidden_size)
        self.loc_generator = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                self.hidden_size,
                weight_attr=weight_attr1,
                bias_attr=bias_attr1),
            nn.Linear(
                self.hidden_size,
                loc_reg_num,
                weight_attr=weight_attr2,
                bias_attr=bias_attr2),
            nn.Sigmoid())

    def forward(self, inputs, targets=None):
        fea = inputs[-1]
        batch_size = fea.shape[0]
        # reshape
        fea = paddle.reshape(fea, [fea.shape[0], fea.shape[1], -1])
        fea = fea.transpose([0, 2, 1])  # (NTC)(batch, width, channels)

        hidden = paddle.zeros((batch_size, self.hidden_size))
        structure_preds = paddle.zeros(
            (batch_size, self.max_text_length + 1, self.num_embeddings))
        loc_preds = paddle.zeros(
            (batch_size, self.max_text_length + 1, self.loc_reg_num))
        structure_preds.stop_gradient = True
        loc_preds.stop_gradient = True
        if self.training and targets is not None:
            structure = targets[0]
            for i in range(self.max_text_length + 1):
                hidden, structure_step, loc_step = self._decode(structure[:, i],
                                                                fea, hidden)
                structure_preds[:, i, :] = structure_step
                loc_preds[:, i, :] = loc_step
        else:
            pre_chars = paddle.zeros(shape=[batch_size], dtype="int32")
            max_text_length = paddle.to_tensor(self.max_text_length)
            # for export
            loc_step, structure_step = None, None
            for i in range(max_text_length + 1):
                hidden, structure_step, loc_step = self._decode(pre_chars, fea,
                                                                hidden)
                pre_chars = structure_step.argmax(axis=1, dtype="int32")
                structure_preds[:, i, :] = structure_step
                loc_preds[:, i, :] = loc_step
        if not self.training:
            structure_preds = F.softmax(structure_preds)
        return {'structure_probs': structure_preds, 'loc_preds': loc_preds}

    def _decode(self, pre_chars, features, hidden):
        """
        Predict table label and coordinates for each step
        @param pre_chars: Table label in previous step
        @param features:
        @param hidden: hidden status in previous step
        @return:
        """
        emb_feature = self.emb(pre_chars)
        # output shape is b * self.hidden_size
        (output, hidden), alpha = self.structure_attention_cell(
            hidden, features, emb_feature)

        # structure
        structure_step = self.structure_generator(output)
        # loc
        loc_step = self.loc_generator(output)
        return hidden, structure_step, loc_step

    def _char_to_onehot(self, input_char):
        input_ont_hot = F.one_hot(input_char, self.num_embeddings)
        return input_ont_hot
