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

import math

import paddle
from paddle import ParamAttr, nn
from paddle.nn import functional as F


def get_para_bias_attr(l2_decay, k):
    regularizer = paddle.regularizer.L2Decay(l2_decay)
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = nn.initializer.Uniform(-stdv, stdv)
    weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    return [weight_attr, bias_attr]


class CTCHead(nn.Layer):
    def __init__(self, in_channels, out_channels, fc_decay=0.0004, **kwargs):
        super(CTCHead, self).__init__()
        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=fc_decay, k=in_channels)
        self.fc = nn.Linear(
            in_channels,
            out_channels,
            weight_attr=weight_attr,
            bias_attr=bias_attr)
        self.out_channels = out_channels

    def forward(self, x, targets=None):
        predicts = self.fc(x)
        if not self.training:
            predicts = F.softmax(predicts, axis=2)
        return predicts
