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
https://github.com/FangShancheng/ABINet/tree/main/modules
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import math

__all__ = ["ResNet45"]


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=1,
        weight_attr=ParamAttr(initializer=KaimingNormal()),
        bias_attr=False)


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2D(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=stride,
        padding=1,
        weight_attr=ParamAttr(initializer=KaimingNormal()),
        bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2D(channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2D(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet45(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 block=BasicBlock,
                 layers=[3, 4, 6, 6, 3],
                 strides=[2, 1, 2, 1, 1]):
        self.inplanes = 32
        super(ResNet45, self).__init__()
        self.conv1 = nn.Conv2D(
            in_channels,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)
        self.bn1 = nn.BatchNorm2D(32)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(block, 32, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=strides[3])
        self.layer5 = self._make_layer(block, 512, layers[4], stride=strides[4])
        self.out_channels = 512

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = True
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    weight_attr=ParamAttr(initializer=KaimingNormal()),
                    bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
