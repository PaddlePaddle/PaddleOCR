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

class TableAttentionHead(nn.Layer):
    def __init__(self, in_channels, hidden_size, loc_type, in_max_len=488, **kwargs):
        super(TableAttentionHead, self).__init__()
        self.input_size = in_channels[-1]
        self.hidden_size = hidden_size
        self.char_num = 280
        self.elem_num = 30

        self.structure_attention_cell = AttentionGRUCell(
            self.input_size, hidden_size, self.elem_num, use_gru=False)
        self.structure_generator = nn.Linear(hidden_size, self.elem_num)
        self.loc_type = loc_type
        self.in_max_len = in_max_len
        
        if self.loc_type == 1:
            self.loc_generator = nn.Linear(hidden_size, 4)
        else:
            if self.in_max_len == 640:
                self.loc_fea_trans = nn.Linear(400, 801)
            elif self.in_max_len == 800:
                self.loc_fea_trans = nn.Linear(625, 801)
            else:
                self.loc_fea_trans = nn.Linear(256, 801)
            self.loc_generator = nn.Linear(self.input_size + hidden_size, 4)
            
    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, mode='Train'):
        # if and else branch are both needed when you want to assign a variable
        # if you modify the var in just one branch, then the modification will not work.
        fea = inputs[-1]
        if len(fea.shape) == 3:
            pass
        else:
            last_shape = int(np.prod(fea.shape[2:])) # gry added
            fea = paddle.reshape(fea, [fea.shape[0], fea.shape[1], last_shape])
            fea = fea.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
        batch_size = fea.shape[0]
        #sp_tokens = targets[2].numpy()
        #char_beg_idx, char_end_idx = sp_tokens[0, 0:2]
        #elem_beg_idx, elem_end_idx = sp_tokens[0, 2:4]
        #elem_char_idx1, elem_char_idx2 = sp_tokens[0, 4:6]
        #max_text_length, max_elem_length, max_cell_num = sp_tokens[0, 6:9]
        max_text_length, max_elem_length, max_cell_num = 100, 800, 500
        
        hidden = paddle.zeros((batch_size, self.hidden_size))
        output_hiddens = []
        if mode == 'Train' and targets is not None:
            structure = targets[0]
            for i in range(max_elem_length+1):
                elem_onehots = self._char_to_onehot(
                    structure[:, i], onehot_dim=self.elem_num)
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
            max_elem_length = paddle.to_tensor(max_elem_length)
            i = 0
            while i < max_elem_length+1:
                elem_onehots = self._char_to_onehot(
                    temp_elem, onehot_dim=self.elem_num)
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
        return {'structure_probs':structure_probs, 'loc_preds':loc_preds}

class AttentionGRUCell(nn.Layer):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionGRUCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias_attr=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias_attr=False)
        self.rnn = nn.GRUCell(
            input_size=input_size + num_embeddings, hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = paddle.unsqueeze(self.h2h(prev_hidden), axis=1)
        res = paddle.add(batch_H_proj, prev_hidden_proj)
        res = paddle.tanh(res)
        e = self.score(res)
        alpha = F.softmax(e, axis=1)
        alpha = paddle.transpose(alpha, [0, 2, 1])
        context = paddle.squeeze(paddle.mm(alpha, batch_H), axis=1)
        concat_context = paddle.concat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha


class AttentionLSTM(nn.Layer):
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionLSTM, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        self.attention_cell = AttentionLSTMCell(
            in_channels, hidden_size, out_channels, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=25):
        batch_size = inputs.shape[0]
        num_steps = batch_max_length

        hidden = (paddle.zeros((batch_size, self.hidden_size)), paddle.zeros(
            (batch_size, self.hidden_size)))
        output_hiddens = []

        if targets is not None:
            for i in range(num_steps):
                # one-hot vectors for a i-th char
                char_onehots = self._char_to_onehot(
                    targets[:, i], onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs,
                                                    char_onehots)

                hidden = (hidden[1][0], hidden[1][1])
                output_hiddens.append(paddle.unsqueeze(hidden[0], axis=1))
            output = paddle.concat(output_hiddens, axis=1)
            probs = self.generator(output)

        else:
            targets = paddle.zeros(shape=[batch_size], dtype="int32")
            probs = None

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs,
                                                    char_onehots)
                probs_step = self.generator(hidden[0])
                hidden = (hidden[1][0], hidden[1][1])
                if probs is None:
                    probs = paddle.unsqueeze(probs_step, axis=1)
                else:
                    probs = paddle.concat(
                        [probs, paddle.unsqueeze(
                            probs_step, axis=1)], axis=1)

                next_input = probs_step.argmax(axis=1)

                targets = next_input

        return probs


class AttentionLSTMCell(nn.Layer):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionLSTMCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias_attr=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias_attr=False)
        if not use_gru:
            self.rnn = nn.LSTMCell(
                input_size=input_size + num_embeddings, hidden_size=hidden_size)
        else:
            self.rnn = nn.GRUCell(
                input_size=input_size + num_embeddings, hidden_size=hidden_size)

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = paddle.unsqueeze(self.h2h(prev_hidden[0]), axis=1)
        res = paddle.add(batch_H_proj, prev_hidden_proj)
        res = paddle.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, axis=1)
        alpha = paddle.transpose(alpha, [0, 2, 1])
        context = paddle.squeeze(paddle.mm(alpha, batch_H), axis=1)
        concat_context = paddle.concat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha
