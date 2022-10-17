# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
https://github.com/LBH1024/CAN/models/densenet.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Bottleneck(nn.Layer):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2D(interChannels)
        self.conv1 = nn.Conv2D(
            nChannels, interChannels, kernel_size=1,
            bias_attr=None)  # Xavier initialization
        self.bn2 = nn.BatchNorm2D(growthRate)
        self.conv2 = nn.Conv2D(
            interChannels, growthRate, kernel_size=3, padding=1,
            bias_attr=None)  # Xavier initialization
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        if self.use_dropout:
            out = self.dropout(out)
        out = paddle.concat([x, out], 1)
        return out


class SingleLayer(nn.Layer):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2D(nChannels)
        self.conv1 = nn.Conv2D(
            nChannels, growthRate, kernel_size=3, padding=1, bias_attr=False)

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x))
        if self.use_dropout:
            out = self.dropout(out)

        out = paddle.concat([x, out], 1)
        return out


class Transition(nn.Layer):
    def __init__(self, nChannels, out_channels, use_dropout):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.conv1 = nn.Conv2D(
            nChannels, out_channels, kernel_size=1, bias_attr=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True, exclusive=False)
        return out


class DenseNet(nn.Layer):
    def __init__(self, growthRate, reduction, bottleneck, use_dropout,
                 input_channel, **kwargs):
        super(DenseNet, self).__init__()

        nDenseBlocks = 16
        nChannels = 2 * growthRate

        self.conv1 = nn.Conv2D(
            input_channel,
            nChannels,
            kernel_size=7,
            padding=3,
            stride=2,
            bias_attr=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        out_channels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, out_channels, use_dropout)

        nChannels = out_channels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        out_channels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, out_channels, use_dropout)

        nChannels = out_channels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck, use_dropout)
        self.out_channels = out_channels

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck,
                    use_dropout):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, use_dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate, use_dropout))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, inputs):
        x, x_m, y = inputs
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        return out, x_m, y
