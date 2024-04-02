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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform

import math

from paddle.vision.ops import DeformConv2D
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Normal, Constant, XavierUniform
from .det_resnet_vd import DeformableConvV2, ConvBNLayer


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 is_dcn=False):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=1,
            act="relu", )
        self.conv1 = ConvBNLayer(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            act="relu",
            is_dcn=is_dcn,
            dcn_groups=1, )
        self.conv2 = ConvBNLayer(
            in_channels=num_filters,
            out_channels=num_filters * 4,
            kernel_size=1,
            act=None, )

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=num_channels,
                out_channels=num_filters * 4,
                kernel_size=1,
                stride=stride, )

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            act="relu")
        self.conv1 = ConvBNLayer(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=num_channels,
                out_channels=num_filters,
                kernel_size=1,
                stride=stride)

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv1)
        y = F.relu(y)
        return y


class ResNet(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 layers=50,
                 out_indices=None,
                 dcn_stage=None):
        super(ResNet, self).__init__()

        self.layers = layers
        self.input_image_channel = in_channels

        supported_layers = [18, 34, 50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_channels = [64, 256, 512,
                        1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.dcn_stage = dcn_stage if dcn_stage is not None else [
            False, False, False, False
        ]
        self.out_indices = out_indices if out_indices is not None else [
            0, 1, 2, 3
        ]

        self.conv = ConvBNLayer(
            in_channels=self.input_image_channel,
            out_channels=64,
            kernel_size=7,
            stride=2,
            act="relu", )
        self.pool2d_max = MaxPool2D(
            kernel_size=3,
            stride=2,
            padding=1, )

        self.stages = []
        self.out_channels = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                is_dcn = self.dcn_stage[block]
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    bottleneck_block = self.add_sublayer(
                        conv_name,
                        BottleneckBlock(
                            num_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            num_filters=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            is_dcn=is_dcn))
                    block_list.append(bottleneck_block)
                    shortcut = True
                if block in self.out_indices:
                    self.out_channels.append(num_filters[block] * 4)
                self.stages.append(nn.Sequential(*block_list))
        else:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        conv_name,
                        BasicBlock(
                            num_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            num_filters=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut))
                    block_list.append(basic_block)
                    shortcut = True
                if block in self.out_indices:
                    self.out_channels.append(num_filters[block])
                self.stages.append(nn.Sequential(*block_list))

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        out = []
        for i, block in enumerate(self.stages):
            y = block(y)
            if i in self.out_indices:
                out.append(y)
        return out
