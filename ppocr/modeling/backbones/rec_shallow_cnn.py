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
https://github.com/open-mmlab/mmocr/blob/1.x/mmocr/models/textrecog/backbones/shallow_cnn.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import MaxPool2D
from paddle.nn.initializer import KaimingNormal, Uniform, Constant


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 num_groups=1):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)

        self.bn = nn.BatchNorm2D(
            num_filters,
            weight_attr=ParamAttr(initializer=Uniform(0, 1)),
            bias_attr=ParamAttr(initializer=Constant(0)))
        self.relu = nn.ReLU()

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.bn(y)
        y = self.relu(y)
        return y


class ShallowCNN(nn.Layer):
    def __init__(self, in_channels=1, hidden_dim=512):
        super().__init__()
        assert isinstance(in_channels, int)
        assert isinstance(hidden_dim, int)

        self.conv1 = ConvBNLayer(
            in_channels, 3, hidden_dim // 2, stride=1, padding=1)
        self.conv2 = ConvBNLayer(
            hidden_dim // 2, 3, hidden_dim, stride=1, padding=1)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.out_channels = hidden_dim

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)

        return x
