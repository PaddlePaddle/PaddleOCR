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
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/layers/conv_layer.py
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/backbones/resnet31_ocr.py
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

__all__ = ["ResNet31", "ResNet45"]


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
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


class ResNet31(nn.Layer):
    '''
    Args:
        in_channels (int): Number of channels of input image tensor.
        layers (list[int]): List of BasicBlock number for each stage.
        channels (list[int]): List of out_channels of Conv2d layer.
        out_indices (None | Sequence[int]): Indices of output stages.
        last_stage_pool (bool): If True, add `MaxPool2d` layer to last stage.
    '''

    def __init__(self,
                 in_channels=3,
                 layers=[1, 2, 5, 3],
                 channels=[64, 128, 256, 256, 512, 512, 512],
                 out_indices=None,
                 last_stage_pool=False):
        super(ResNet31, self).__init__()
        assert isinstance(in_channels, int)
        assert isinstance(last_stage_pool, bool)

        self.out_indices = out_indices
        self.last_stage_pool = last_stage_pool

        # conv 1 (Conv Conv)
        self.conv1_1 = nn.Conv2D(
            in_channels, channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2D(channels[0])
        self.relu1_1 = nn.ReLU()

        self.conv1_2 = nn.Conv2D(
            channels[0], channels[1], kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2D(channels[1])
        self.relu1_2 = nn.ReLU()

        # conv 2 (Max-pooling, Residual block, Conv)
        self.pool2 = nn.MaxPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block2 = self._make_layer(channels[1], channels[2], layers[0])
        self.conv2 = nn.Conv2D(
            channels[2], channels[2], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(channels[2])
        self.relu2 = nn.ReLU()

        # conv 3 (Max-pooling, Residual block, Conv)
        self.pool3 = nn.MaxPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block3 = self._make_layer(channels[2], channels[3], layers[1])
        self.conv3 = nn.Conv2D(
            channels[3], channels[3], kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2D(channels[3])
        self.relu3 = nn.ReLU()

        # conv 4 (Max-pooling, Residual block, Conv)
        self.pool4 = nn.MaxPool2D(
            kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=True)
        self.block4 = self._make_layer(channels[3], channels[4], layers[2])
        self.conv4 = nn.Conv2D(
            channels[4], channels[4], kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2D(channels[4])
        self.relu4 = nn.ReLU()

        # conv 5 ((Max-pooling), Residual block, Conv)
        self.pool5 = None
        if self.last_stage_pool:
            self.pool5 = nn.MaxPool2D(
                kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block5 = self._make_layer(channels[4], channels[5], layers[3])
        self.conv5 = nn.Conv2D(
            channels[5], channels[5], kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2D(channels[5])
        self.relu5 = nn.ReLU()

        self.out_channels = channels[-1]

    def _make_layer(self, input_channels, output_channels, blocks):
        layers = []
        for _ in range(blocks):
            downsample = None
            if input_channels != output_channels:
                downsample = nn.Sequential(
                    nn.Conv2D(
                        input_channels,
                        output_channels,
                        kernel_size=1,
                        stride=1,
                        weight_attr=ParamAttr(initializer=KaimingNormal()),
                        bias_attr=False),
                    nn.BatchNorm2D(output_channels), )

            layers.append(
                BasicBlock(
                    input_channels, output_channels, downsample=downsample))
            input_channels = output_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        outs = []
        for i in range(4):
            layer_index = i + 2
            pool_layer = getattr(self, f'pool{layer_index}')
            block_layer = getattr(self, f'block{layer_index}')
            conv_layer = getattr(self, f'conv{layer_index}')
            bn_layer = getattr(self, f'bn{layer_index}')
            relu_layer = getattr(self, f'relu{layer_index}')

            if pool_layer is not None:
                x = pool_layer(x)
            x = block_layer(x)
            x = conv_layer(x)
            x = bn_layer(x)
            x = relu_layer(x)

            outs.append(x)

        if self.out_indices is not None:
            return tuple([outs[i] for i in self.out_indices])

        return x


class ResNet(nn.Layer):
    def __init__(self, block, layers, in_channels=3):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2D(
            3,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)
        self.bn1 = nn.BatchNorm2D(32)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=1)
        self.out_channels = 512

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2D):
        #         n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))

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
        # print(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


def ResNet45(in_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 6, 3], in_channels=in_channels)
