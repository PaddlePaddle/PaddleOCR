# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
https://github.com/hikopensource/DAVAR-Lab-OCR/blob/main/davarocr/davar_common/models/loss/cross_entropy_loss.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn

from .basic_loss import CELoss, DistanceLoss


class RFLLoss(nn.Layer):
    def __init__(self, ignore_index=-100, **kwargs):
        super().__init__()

        self.cnt_loss = nn.MSELoss(**kwargs)
        self.seq_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predicts, batch):

        self.total_loss = {}
        total_loss = 0.0
        if isinstance(predicts, tuple) or isinstance(predicts, list):
            cnt_outputs, seq_outputs = predicts
        else:
            cnt_outputs, seq_outputs = predicts, None
        # batch [image, label, length, cnt_label]
        if cnt_outputs is not None:
            cnt_loss = self.cnt_loss(cnt_outputs,
                                     paddle.cast(batch[3], paddle.float32))
            self.total_loss['cnt_loss'] = cnt_loss
            total_loss += cnt_loss

        if seq_outputs is not None:
            targets = batch[1].astype("int64")
            label_lengths = batch[2].astype('int64')
            batch_size, num_steps, num_classes = seq_outputs.shape[
                0], seq_outputs.shape[1], seq_outputs.shape[2]
            assert len(targets.shape) == len(list(seq_outputs.shape)) - 1, \
                "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

            inputs = seq_outputs[:, :-1, :]
            targets = targets[:, 1:]

            inputs = paddle.reshape(inputs, [-1, inputs.shape[-1]])
            targets = paddle.reshape(targets, [-1])
            seq_loss = self.seq_loss(inputs, targets)
            self.total_loss['seq_loss'] = seq_loss
            total_loss += seq_loss

        self.total_loss['loss'] = total_loss
        return self.total_loss
