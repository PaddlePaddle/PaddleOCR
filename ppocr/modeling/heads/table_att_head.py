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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from .rec_att_head import AttentionGRUCell


class TableAttentionHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 loc_type,
                 in_max_len=488,
                 max_text_length=800,
                 out_channels=30,
                 point_num=2,
                 **kwargs):
        super(TableAttentionHead, self).__init__()
        self.input_size = in_channels[-1]
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.max_text_length = max_text_length

        self.structure_attention_cell = AttentionGRUCell(
            self.input_size, hidden_size, self.out_channels, use_gru=False)
        self.structure_generator = nn.Linear(hidden_size, self.out_channels)
        self.loc_type = loc_type
        self.in_max_len = in_max_len

        if self.loc_type == 1:
            self.loc_generator = nn.Linear(hidden_size, 4)
        else:
            if self.in_max_len == 640:
                self.loc_fea_trans = nn.Linear(400, self.max_text_length + 1)
            elif self.in_max_len == 800:
                self.loc_fea_trans = nn.Linear(625, self.max_text_length + 1)
            else:
                self.loc_fea_trans = nn.Linear(256, self.max_text_length + 1)
            self.loc_generator = nn.Linear(self.input_size + hidden_size,
                                           point_num * 2)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None):
        # if and else branch are both needed when you want to assign a variable
        # if you modify the var in just one branch, then the modification will not work.
        fea = inputs[-1]
        if len(fea.shape) == 3:
            pass
        else:
            last_shape = int(np.prod(fea.shape[2:]))  # gry added
            fea = paddle.reshape(fea, [fea.shape[0], fea.shape[1], last_shape])
            fea = fea.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
        batch_size = fea.shape[0]

        hidden = paddle.zeros((batch_size, self.hidden_size))
        output_hiddens = []
        if self.training and targets is not None:
            structure = targets[0]
            for i in range(self.max_text_length + 1):
                elem_onehots = self._char_to_onehot(
                    structure[:, i], onehot_dim=self.out_channels)
                (outputs, hidden), alpha = self.structure_attention_cell(
                    hidden, fea, elem_onehots)
                output_hiddens.append(paddle.unsqueeze(outputs, axis=1))
            output = paddle.concat(output_hiddens, axis=1)
            structure_probs = self.structure_generator(output)
            if self.loc_type == 1:
                loc_preds = self.loc_generator(output)
                loc_preds = F.sigmoid(loc_preds)
            else:
                loc_fea = fea.transpose([0, 2, 1])
                loc_fea = self.loc_fea_trans(loc_fea)
                loc_fea = loc_fea.transpose([0, 2, 1])
                loc_concat = paddle.concat([output, loc_fea], axis=2)
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
            i = 0
            while i < max_text_length + 1:
                elem_onehots = self._char_to_onehot(
                    temp_elem, onehot_dim=self.out_channels)
                (outputs, hidden), alpha = self.structure_attention_cell(
                    hidden, fea, elem_onehots)
                output_hiddens.append(paddle.unsqueeze(outputs, axis=1))
                structure_probs_step = self.structure_generator(outputs)
                temp_elem = structure_probs_step.argmax(axis=1, dtype="int32")
                i += 1

            output = paddle.concat(output_hiddens, axis=1)
            structure_probs = self.structure_generator(output)
            structure_probs = F.softmax(structure_probs)
            if self.loc_type == 1:
                loc_preds = self.loc_generator(output)
                loc_preds = F.sigmoid(loc_preds)
            else:
                loc_fea = fea.transpose([0, 2, 1])
                loc_fea = self.loc_fea_trans(loc_fea)
                loc_fea = loc_fea.transpose([0, 2, 1])
                loc_concat = paddle.concat([output, loc_fea], axis=2)
                loc_preds = self.loc_generator(loc_concat)
                loc_preds = F.sigmoid(loc_preds)
        return {'structure_probs': structure_probs, 'loc_preds': loc_preds}
