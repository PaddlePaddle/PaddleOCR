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
https://github.com/hikopensource/DAVAR-Lab-OCR/blob/main/davarocr/davar_rcg/models/backbones/ResNetRFL.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn

from paddle.nn.initializer import TruncatedNormal, Constant, Normal, KaimingNormal

kaiming_init_ = KaimingNormal()
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


class BasicBlock(nn.Layer):
    """Res-net Basic Block"""

    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, norm_type="BN", **kwargs
    ):
        """
        Args:
            inplanes (int): input channel
            planes (int): channels of the middle feature
            stride (int): stride of the convolution
            downsample (int): type of the down_sample
            norm_type (str): type of the normalization
            **kwargs (None): backup parameter
        """
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm(planes)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias_attr=False,
        )

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


class ResNetRFL(nn.Layer):
    def __init__(self, in_channels, out_channels=512, use_cnt=True, use_seq=True):
        """

        Args:
            in_channels (int): input channel
            out_channels (int): output channel
        """
        super(ResNetRFL, self).__init__()
        assert use_cnt or use_seq
        self.use_cnt, self.use_seq = use_cnt, use_seq
        self.backbone = RFLBase(in_channels)

        self.out_channels = out_channels
        self.out_channels_block = [
            int(self.out_channels / 4),
            int(self.out_channels / 2),
            self.out_channels,
            self.out_channels,
        ]
        block = BasicBlock
        layers = [1, 2, 5, 3]
        self.inplanes = int(self.out_channels // 2)

        self.relu = nn.ReLU()
        if self.use_seq:
            self.maxpool3 = nn.MaxPool2D(kernel_size=2, stride=(2, 1), padding=(0, 1))
            self.layer3 = self._make_layer(
                block, self.out_channels_block[2], layers[2], stride=1
            )
            self.conv3 = nn.Conv2D(
                self.out_channels_block[2],
                self.out_channels_block[2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False,
            )
            self.bn3 = nn.BatchNorm(self.out_channels_block[2])

            self.layer4 = self._make_layer(
                block, self.out_channels_block[3], layers[3], stride=1
            )
            self.conv4_1 = nn.Conv2D(
                self.out_channels_block[3],
                self.out_channels_block[3],
                kernel_size=2,
                stride=(2, 1),
                padding=(0, 1),
                bias_attr=False,
            )
            self.bn4_1 = nn.BatchNorm(self.out_channels_block[3])
            self.conv4_2 = nn.Conv2D(
                self.out_channels_block[3],
                self.out_channels_block[3],
                kernel_size=2,
                stride=1,
                padding=0,
                bias_attr=False,
            )
            self.bn4_2 = nn.BatchNorm(self.out_channels_block[3])

        if self.use_cnt:
            self.inplanes = int(self.out_channels // 2)
            self.v_maxpool3 = nn.MaxPool2D(kernel_size=2, stride=(2, 1), padding=(0, 1))
            self.v_layer3 = self._make_layer(
                block, self.out_channels_block[2], layers[2], stride=1
            )
            self.v_conv3 = nn.Conv2D(
                self.out_channels_block[2],
                self.out_channels_block[2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False,
            )
            self.v_bn3 = nn.BatchNorm(self.out_channels_block[2])

            self.v_layer4 = self._make_layer(
                block, self.out_channels_block[3], layers[3], stride=1
            )
            self.v_conv4_1 = nn.Conv2D(
                self.out_channels_block[3],
                self.out_channels_block[3],
                kernel_size=2,
                stride=(2, 1),
                padding=(0, 1),
                bias_attr=False,
            )
            self.v_bn4_1 = nn.BatchNorm(self.out_channels_block[3])
            self.v_conv4_2 = nn.Conv2D(
                self.out_channels_block[3],
                self.out_channels_block[3],
                kernel_size=2,
                stride=1,
                padding=0,
                bias_attr=False,
            )
            self.v_bn4_2 = nn.BatchNorm(self.out_channels_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False,
                ),
                nn.BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x_1 = self.backbone(inputs)

        if self.use_cnt:
            v_x = self.v_maxpool3(x_1)
            v_x = self.v_layer3(v_x)
            v_x = self.v_conv3(v_x)
            v_x = self.v_bn3(v_x)
            visual_feature_2 = self.relu(v_x)

            v_x = self.v_layer4(visual_feature_2)
            v_x = self.v_conv4_1(v_x)
            v_x = self.v_bn4_1(v_x)
            v_x = self.relu(v_x)
            v_x = self.v_conv4_2(v_x)
            v_x = self.v_bn4_2(v_x)
            visual_feature_3 = self.relu(v_x)
        else:
            visual_feature_3 = None
        if self.use_seq:
            x = self.maxpool3(x_1)
            x = self.layer3(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x_2 = self.relu(x)

            x = self.layer4(x_2)
            x = self.conv4_1(x)
            x = self.bn4_1(x)
            x = self.relu(x)
            x = self.conv4_2(x)
            x = self.bn4_2(x)
            x_3 = self.relu(x)
        else:
            x_3 = None

        return [visual_feature_3, x_3]


class ResNetBase(nn.Layer):
    def __init__(self, in_channels, out_channels, block, layers):
        super(ResNetBase, self).__init__()

        self.out_channels_block = [
            int(out_channels / 4),
            int(out_channels / 2),
            out_channels,
            out_channels,
        ]

        self.inplanes = int(out_channels / 8)
        self.conv0_1 = nn.Conv2D(
            in_channels,
            int(out_channels / 16),
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
        )
        self.bn0_1 = nn.BatchNorm(int(out_channels / 16))
        self.conv0_2 = nn.Conv2D(
            int(out_channels / 16),
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
        )
        self.bn0_2 = nn.BatchNorm(self.inplanes)
        self.relu = nn.ReLU()

        self.maxpool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.out_channels_block[0], layers[0])
        self.conv1 = nn.Conv2D(
            self.out_channels_block[0],
            self.out_channels_block[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
        )
        self.bn1 = nn.BatchNorm(self.out_channels_block[0])

        self.maxpool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(
            block, self.out_channels_block[1], layers[1], stride=1
        )
        self.conv2 = nn.Conv2D(
            self.out_channels_block[1],
            self.out_channels_block[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
        )
        self.bn2 = nn.BatchNorm(self.out_channels_block[1])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False,
                ),
                nn.BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class RFLBase(nn.Layer):
    """Reciprocal feature learning share backbone network"""

    def __init__(self, in_channels, out_channels=512):
        super(RFLBase, self).__init__()
        self.ConvNet = ResNetBase(in_channels, out_channels, BasicBlock, [1, 2, 5, 3])

    def forward(self, inputs):
        return self.ConvNet(inputs)
