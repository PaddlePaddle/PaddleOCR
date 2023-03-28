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
"""
This code is refer from:
https://github.com/LBH1024/CAN/models/can.py
"""

import paddle
import paddle.nn as nn
import numpy as np


class CANLoss(nn.Layer):
    '''
    CANLoss is consist of two part:
        word_average_loss: average accuracy of the symbol
        counting_loss: counting loss of every symbol
    '''

    def __init__(self):
        super(CANLoss, self).__init__()

        self.use_label_mask = False
        self.out_channel = 111
        self.cross = nn.CrossEntropyLoss(
            reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()
        self.counting_loss = nn.SmoothL1Loss(reduction='mean')
        self.ratio = 16

    def forward(self, preds, batch):
        word_probs = preds[0]
        counting_preds = preds[1]
        counting_preds1 = preds[2]
        counting_preds2 = preds[3]
        labels = batch[2]
        labels_mask = batch[3]
        counting_labels = gen_counting_label(labels, self.out_channel, True)
        counting_loss = self.counting_loss(counting_preds1, counting_labels) + self.counting_loss(counting_preds2, counting_labels) \
                        + self.counting_loss(counting_preds, counting_labels)

        word_loss = self.cross(
            paddle.reshape(word_probs, [-1, word_probs.shape[-1]]),
            paddle.reshape(labels, [-1]))
        word_average_loss = paddle.sum(
            paddle.reshape(word_loss * labels_mask, [-1])) / (
                paddle.sum(labels_mask) + 1e-10
            ) if self.use_label_mask else word_loss
        loss = word_average_loss + counting_loss
        return {'loss': loss}


def gen_counting_label(labels, channel, tag):
    b, t = labels.shape
    counting_labels = np.zeros([b, channel])

    if tag:
        ignore = [0, 1, 107, 108, 109, 110]
    else:
        ignore = []
    for i in range(b):
        for j in range(t):
            k = labels[i][j]
            if k in ignore:
                continue
            else:
                counting_labels[i][k] += 1
    counting_labels = paddle.to_tensor(counting_labels, dtype='float32')
    return counting_labels
