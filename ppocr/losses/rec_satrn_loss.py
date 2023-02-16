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
https://github.com/open-mmlab/mmocr/blob/1.x/mmocr/models/textrecog/module_losses/ce_module_loss.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn


class SATRNLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(SATRNLoss, self).__init__()
        ignore_index = kwargs.get('ignore_index', 92)  # 6626
        self.loss_func = paddle.nn.loss.CrossEntropyLoss(
            reduction="none", ignore_index=ignore_index)

    def forward(self, predicts, batch):
        predict = predicts[:, :
                           -1, :]  # ignore last index of outputs to be in same seq_len with targets
        label = batch[1].astype(
            "int64")[:, 1:]  # ignore first index of target in loss calculation
        batch_size, num_steps, num_classes = predict.shape[0], predict.shape[
            1], predict.shape[2]
        assert len(label.shape) == len(list(predict.shape)) - 1, \
            "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        inputs = paddle.reshape(predict, [-1, num_classes])
        targets = paddle.reshape(label, [-1])
        loss = self.loss_func(inputs, targets)
        return {'loss': loss.mean()}
