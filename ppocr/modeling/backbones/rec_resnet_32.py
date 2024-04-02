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
https://github.com/hikopensource/DAVAR-Lab-OCR/davarocr/davar_rcg/models/backbones/ResNet32.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.nn as nn

__all__ = ["ResNet32"]

conv_weight_attr = nn.initializer.KaimingNormal()

class ResNet32(nn.Layer):
    """
    Feature Extractor is proposed in  FAN Ref [1]

    Ref [1]: Focusing Attention: Towards Accurate Text Recognition in Neural Images ICCV-2017
    """

    def __init__(self, in_channels, out_channels=512):
        """

        Args:
            in_channels (int): input channel
            output_channel (int): output channel
        """
        super(ResNet32, self).__init__()
        self.out_channels = out_channels
        self.ConvNet = ResNet(in_channels, out_channels, BasicBlock, [1, 2, 5, 3])

    def forward(self, inputs):
        """
        Args:
            inputs: input feature

        Returns:
            output feature

        """
        return self.ConvNet(inputs)

class BasicBlock(nn.Layer):
    """Res-net Basic Block"""
    expansion = 1

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None,
                 norm_type='BN', **kwargs):
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
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        """

        Args:
            in_planes (int): input channel
            out_planes (int): channels of the middle feature
            stride (int): stride of the convolution
        Returns:
            nn.Layer: Conv2D with kernel = 3

        """

        return nn.Conv2D(in_planes, out_planes,
                         kernel_size=3, stride=stride,
                         padding=1, weight_attr=conv_weight_attr,
                         bias_attr=False)

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

class ResNet(nn.Layer):
    """Res-Net network structure"""
    def __init__(self, input_channel,
                 output_channel, block, layers):
        """

        Args:
            input_channel (int): input channel
            output_channel (int): output channel
            block (BasicBlock): convolution block
            layers (list): layers of the block
        """
        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4),
                                     int(output_channel / 2),
                                     output_channel,
                                     output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2D(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, 
                                 padding=1, 
                                 weight_attr=conv_weight_attr,
                                 bias_attr=False)
        self.bn0_1 = nn.BatchNorm2D(int(output_channel / 16))
        self.conv0_2 = nn.Conv2D(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1,
                                 padding=1, 
                                 weight_attr=conv_weight_attr,
                                 bias_attr=False)
        self.bn0_2 = nn.BatchNorm2D(self.inplanes)
        self.relu = nn.ReLU()

        self.maxpool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block,
                                       self.output_channel_block[0],
                                       layers[0])
        self.conv1 = nn.Conv2D(self.output_channel_block[0],
                               self.output_channel_block[0],
                               kernel_size=3, stride=1,
                               padding=1, 
                               weight_attr=conv_weight_attr,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block,
                                       self.output_channel_block[1],
                                       layers[1], stride=1)
        self.conv2 = nn.Conv2D(self.output_channel_block[1],
                               self.output_channel_block[1],
                               kernel_size=3, stride=1,
                               padding=1, 
                               weight_attr=conv_weight_attr,
                               bias_attr=False,)
        self.bn2 = nn.BatchNorm2D(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2D(kernel_size=2,
                                     stride=(2, 1),
                                     padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2],
                                       layers[2], stride=1)
        self.conv3 = nn.Conv2D(self.output_channel_block[2],
                               self.output_channel_block[2],
                               kernel_size=3, stride=1,
                               padding=1, 
                               weight_attr=conv_weight_attr,
                               bias_attr=False)
        self.bn3 = nn.BatchNorm2D(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3],
                                       layers[3], stride=1)
        self.conv4_1 = nn.Conv2D(self.output_channel_block[3],
                                 self.output_channel_block[3],
                                 kernel_size=2, stride=(2, 1),
                                 padding=(0, 1), 
                                 weight_attr=conv_weight_attr,
                                 bias_attr=False)
        self.bn4_1 = nn.BatchNorm2D(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2D(self.output_channel_block[3],
                                 self.output_channel_block[3],
                                 kernel_size=2, stride=1,
                                 padding=0, 
                                 weight_attr=conv_weight_attr,
                                 bias_attr=False)
        self.bn4_2 = nn.BatchNorm2D(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        """

        Args:
            block (block): convolution block
            planes (int): input channels
            blocks (list): layers of the block
            stride (int): stride of the convolution

        Returns:
            nn.Sequential: the combination of the convolution block

        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride,
                          weight_attr=conv_weight_attr, 
                          bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
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

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)
        return x
