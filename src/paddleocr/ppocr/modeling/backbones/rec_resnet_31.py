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
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

__all__ = ["ResNet31"]


def conv3x3(in_channel, out_channel, stride=1, conv_weight_attr=None):
    return nn.Conv2D(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=stride,
        padding=1,
        weight_attr=conv_weight_attr,
        bias_attr=False,
    )


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        downsample=False,
        conv_weight_attr=None,
        bn_weight_attr=None,
    ):
        super().__init__()
        self.conv1 = conv3x3(
            in_channels, channels, stride, conv_weight_attr=conv_weight_attr
        )
        self.bn1 = nn.BatchNorm2D(channels, weight_attr=bn_weight_attr)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(channels, channels, conv_weight_attr=conv_weight_attr)
        self.bn2 = nn.BatchNorm2D(channels, weight_attr=bn_weight_attr)
        self.downsample = downsample
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2D(
                    in_channels,
                    channels * self.expansion,
                    1,
                    stride,
                    weight_attr=conv_weight_attr,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(channels * self.expansion, weight_attr=bn_weight_attr),
            )
        else:
            self.downsample = nn.Sequential()
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet31(nn.Layer):
    """
    Args:
        in_channels (int): Number of channels of input image tensor.
        layers (list[int]): List of BasicBlock number for each stage.
        channels (list[int]): List of out_channels of Conv2d layer.
        out_indices (None | Sequence[int]): Indices of output stages.
        last_stage_pool (bool): If True, add `MaxPool2d` layer to last stage.
        init_type (None | str): the config to control the initialization.
    """

    def __init__(
        self,
        in_channels=3,
        layers=[1, 2, 5, 3],
        channels=[64, 128, 256, 256, 512, 512, 512],
        out_indices=None,
        last_stage_pool=False,
        init_type=None,
    ):
        super(ResNet31, self).__init__()
        assert isinstance(in_channels, int)
        assert isinstance(last_stage_pool, bool)

        self.out_indices = out_indices
        self.last_stage_pool = last_stage_pool

        conv_weight_attr = None
        bn_weight_attr = None

        if init_type is not None:
            support_dict = ["KaimingNormal"]
            assert init_type in support_dict, Exception(
                "resnet31 only support {}".format(support_dict)
            )
            conv_weight_attr = nn.initializer.KaimingNormal()
            bn_weight_attr = ParamAttr(
                initializer=nn.initializer.Uniform(), learning_rate=1
            )

        # conv 1 (Conv Conv)
        self.conv1_1 = nn.Conv2D(
            in_channels,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=conv_weight_attr,
        )
        self.bn1_1 = nn.BatchNorm2D(channels[0], weight_attr=bn_weight_attr)
        self.relu1_1 = nn.ReLU()

        self.conv1_2 = nn.Conv2D(
            channels[0],
            channels[1],
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=conv_weight_attr,
        )
        self.bn1_2 = nn.BatchNorm2D(channels[1], weight_attr=bn_weight_attr)
        self.relu1_2 = nn.ReLU()

        # conv 2 (Max-pooling, Residual block, Conv)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block2 = self._make_layer(
            channels[1],
            channels[2],
            layers[0],
            conv_weight_attr=conv_weight_attr,
            bn_weight_attr=bn_weight_attr,
        )
        self.conv2 = nn.Conv2D(
            channels[2],
            channels[2],
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=conv_weight_attr,
        )
        self.bn2 = nn.BatchNorm2D(channels[2], weight_attr=bn_weight_attr)
        self.relu2 = nn.ReLU()

        # conv 3 (Max-pooling, Residual block, Conv)
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block3 = self._make_layer(
            channels[2],
            channels[3],
            layers[1],
            conv_weight_attr=conv_weight_attr,
            bn_weight_attr=bn_weight_attr,
        )
        self.conv3 = nn.Conv2D(
            channels[3],
            channels[3],
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=conv_weight_attr,
        )
        self.bn3 = nn.BatchNorm2D(channels[3], weight_attr=bn_weight_attr)
        self.relu3 = nn.ReLU()

        # conv 4 (Max-pooling, Residual block, Conv)
        self.pool4 = nn.MaxPool2D(
            kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=True
        )
        self.block4 = self._make_layer(
            channels[3],
            channels[4],
            layers[2],
            conv_weight_attr=conv_weight_attr,
            bn_weight_attr=bn_weight_attr,
        )
        self.conv4 = nn.Conv2D(
            channels[4],
            channels[4],
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=conv_weight_attr,
        )
        self.bn4 = nn.BatchNorm2D(channels[4], weight_attr=bn_weight_attr)
        self.relu4 = nn.ReLU()

        # conv 5 ((Max-pooling), Residual block, Conv)
        self.pool5 = None
        if self.last_stage_pool:
            self.pool5 = nn.MaxPool2D(
                kernel_size=2, stride=2, padding=0, ceil_mode=True
            )
        self.block5 = self._make_layer(
            channels[4],
            channels[5],
            layers[3],
            conv_weight_attr=conv_weight_attr,
            bn_weight_attr=bn_weight_attr,
        )
        self.conv5 = nn.Conv2D(
            channels[5],
            channels[5],
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=conv_weight_attr,
        )
        self.bn5 = nn.BatchNorm2D(channels[5], weight_attr=bn_weight_attr)
        self.relu5 = nn.ReLU()

        self.out_channels = channels[-1]

    def _make_layer(
        self,
        input_channels,
        output_channels,
        blocks,
        conv_weight_attr=None,
        bn_weight_attr=None,
    ):
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
                        weight_attr=conv_weight_attr,
                        bias_attr=False,
                    ),
                    nn.BatchNorm2D(output_channels, weight_attr=bn_weight_attr),
                )

            layers.append(
                BasicBlock(
                    input_channels,
                    output_channels,
                    downsample=downsample,
                    conv_weight_attr=conv_weight_attr,
                    bn_weight_attr=bn_weight_attr,
                )
            )
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
            pool_layer = getattr(self, f"pool{layer_index}")
            block_layer = getattr(self, f"block{layer_index}")
            conv_layer = getattr(self, f"conv{layer_index}")
            bn_layer = getattr(self, f"bn{layer_index}")
            relu_layer = getattr(self, f"relu{layer_index}")

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
