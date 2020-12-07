# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
from paddle.jit import to_static


class AttentionHead(nn.Layer):
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionHead, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        self.attention_cell = AttentionGRUCell(in_channels, hidden_size, out_channels, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=25):
        batch_size = inputs.shape[0]
        num_steps = batch_max_length  # +1 for [s] at end of sentence.

        # output_hiddens = paddle.zeros((batch_size, num_steps, self.hidden_size), dtype="float32")
        # hidden = (paddle.zeros((batch_size, self.hidden_size)), paddle.zeros((batch_size, self.hidden_size)))
        hidden = paddle.zeros((batch_size, self.hidden_size))
        output_hiddens = []

        if targets is not None:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(targets[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs, char_onehots)
                # output_hiddens[:, i, :] = hidden[0]
                # print(hidden[0].shape, hidden[1][0].shape, hidden[1][1].shape)
                # print("hidden.shape: ", outputs.shape, hidden.shape)
                output_hiddens.append(paddle.unsqueeze(outputs, axis=1))
            output = paddle.concat(output_hiddens, axis=1)
            probs = self.generator(output)

        else:
            targets = paddle.zeros(shape=[batch_size], dtype="int32")
            probs = None

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs, char_onehots)
                probs_step = self.generator(outputs)
                probs = paddle.unsqueeze(probs_step, axis=1) if probs is None else paddle.concat([probs, paddle.unsqueeze(probs_step, axis=1)], axis=1)
                # probs[:, i, :] = probs_step
                next_input = probs_step.argmax(axis=1)
                # _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionGRUCell(nn.Layer):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionGRUCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias_attr=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias_attr=False)

        #self.rnn = nn.GRUCell(input_size=input_size+num_embeddings, hidden_size=hidden_size)
        self.rnn = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # print(prev_hidden.shape, batch_H.shape)
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = paddle.unsqueeze(self.h2h(prev_hidden), axis=1)
        # print(batch_H_proj.shape, prev_hidden_proj.shape)
        res = paddle.add(batch_H_proj, prev_hidden_proj)
        res = paddle.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, axis=1)
        alpha = paddle.transpose(alpha, [0, 2, 1])
        context = paddle.squeeze(paddle.mm(alpha, batch_H), axis=1)
        concat_context = paddle.concat([context, char_onehots], 1)
        # print(concat_context.shape, prev_hidden.shape)
        cur_hidden = self.rnn(context, prev_hidden)

        return cur_hidden, alpha

    

class AttentionLSTM(nn.Layer):
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionLSTM, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        self.attention_cell = AttentionLSTMCell(in_channels, hidden_size, out_channels, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=25):
        batch_size = inputs.shape[0]
        num_steps = batch_max_length  # +1 for [s] at end of sentence.

        hidden = (paddle.zeros((batch_size, self.hidden_size)), paddle.zeros((batch_size, self.hidden_size)))
        output_hiddens = []

        if targets is not None:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(targets[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)
                # output_hiddens[:, i, :] = hidden[0]
                # print(hidden[0].shape, hidden[1][0].shape, hidden[1][1].shape)
                hidden = (hidden[1][0], hidden[1][1])
                output_hiddens.append(paddle.unsqueeze(hidden[0], axis=1))
            output = paddle.concat(output_hiddens, axis=1)
            probs = self.generator(output)

        else:
            targets = paddle.zeros(shape=[batch_size], dtype="int32")
            probs = None

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)
                probs_step = self.generator(hidden[0])
                hidden = (hidden[1][0], hidden[1][1])
                probs = paddle.unsqueeze(probs_step, axis=1) if probs is None else paddle.concat([probs, paddle.unsqueeze(probs_step, axis=1)], axis=1)
                # probs[:, i, :] = probs_step
                next_input = probs_step.argmax(axis=1)
                # _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionLSTMCell(nn.Layer):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionLSTMCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias_attr=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias_attr=False)
        if not use_gru:
            self.rnn = nn.LSTMCell(input_size=input_size + num_embeddings, hidden_size=hidden_size)
        else:
            self.rnn = nn.GRUCell(input_size=input_size + num_embeddings, hidden_size=hidden_size)

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = paddle.unsqueeze(self.h2h(prev_hidden[0]), axis=1)
        # print(batch_H_proj.shape, prev_hidden_proj.shape)
        res = paddle.add(batch_H_proj,  prev_hidden_proj)
        res = paddle.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, axis=1)
        alpha = paddle.transpose(alpha, [0, 2, 1])
        context = paddle.squeeze(paddle.mm(alpha, batch_H), axis=1)
        concat_context = paddle.concat([context, char_onehots], 1)
        # print(concat_context.shape, prev_hidden[0].shape, prev_hidden[1].shape)
        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha

    
if __name__ == '__main__':
    paddle.disable_static()

    model = Attention(100, 200, 10)

    x = np.random.uniform(-1, 1, [2, 10, 100]).astype(np.float32)
    y = np.random.randint(0, 10, [2, 21]).astype(np.int32)

    xp = paddle.to_tensor(x)
    yp = paddle.to_tensor(y)

    res = model(inputs=xp, targets=yp, is_train=True, batch_max_length=20)
    print("res: ", res.shape)
    # export_model()
    # x = paddle.uniform(shape=[1, 2, 32, 32, 32], dtype='float32', min=-1, max=1)
    # res = paddle.nn.functional.avg_pool3d(x, 3, 1, 0)
    # print(res.shape)
