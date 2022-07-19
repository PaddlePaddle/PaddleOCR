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
https://github.com/ayumiymk/aster.pytorch/blob/master/lib/models/resnet_aster.py
"""
import paddle
import paddle.nn as nn

import sys
import math
from paddle.nn.initializer import KaimingNormal, Constant

zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias_attr=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(
        in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)


def get_sinusoid_encoding(n_position, feat_dim, wave_length=10000):
    # [n_position]
    positions = paddle.arange(0, n_position)
    # [feat_dim]
    dim_range = paddle.arange(0, feat_dim)
    dim_range = paddle.pow(wave_length, 2 * (dim_range // 2) / feat_dim)
    # [n_position, feat_dim]
    angles = paddle.unsqueeze(
        positions, axis=1) / paddle.unsqueeze(
            dim_range, axis=0)
    angles = paddle.cast(angles, "float32")
    angles[:, 0::2] = paddle.sin(angles[:, 0::2])
    angles[:, 1::2] = paddle.cos(angles[:, 1::2])
    return angles


class AsterBlock(nn.Layer):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AsterBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
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


class ResNet_ASTER(nn.Layer):
    """For aster or crnn"""

    def __init__(self, with_lstm=True, n_group=1, in_channels=3):
        super(ResNet_ASTER, self).__init__()
        self.with_lstm = with_lstm
        self.n_group = n_group

        self.layer0 = nn.Sequential(
            nn.Conv2D(
                in_channels,
                32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias_attr=False),
            nn.BatchNorm2D(32),
            nn.ReLU())

        self.inplanes = 32
        self.layer1 = self._make_layer(32, 3, [2, 2])  # [16, 50]
        self.layer2 = self._make_layer(64, 4, [2, 2])  # [8, 25]
        self.layer3 = self._make_layer(128, 6, [2, 1])  # [4, 25]
        self.layer4 = self._make_layer(256, 6, [2, 1])  # [2, 25]
        self.layer5 = self._make_layer(512, 3, [2, 1])  # [1, 25]

        if with_lstm:
            self.rnn = nn.LSTM(512, 256, direction="bidirect", num_layers=2)
            self.out_channels = 2 * 256
        else:
            self.out_channels = 512

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride), nn.BatchNorm2D(planes))

        layers = []
        layers.append(AsterBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(AsterBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        cnn_feat = x5.squeeze(2)  # [N, c, w]
        cnn_feat = paddle.transpose(cnn_feat, perm=[0, 2, 1])
        if self.with_lstm:
            rnn_feat, _ = self.rnn(cnn_feat)
            return rnn_feat
        else:
            return cnn_feat


class Block(nn.Layer):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Block, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2D(planes)
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
    def __init__(self, in_channels=3, compress_layer=False):
        super(ResNet45, self).__init__()
        self.compress_layer = compress_layer

        self.conv1_new = nn.Conv2D(
            in_channels,
            32,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias_attr=False)
        self.bn1 = nn.BatchNorm2D(32)
        self.relu = nn.ReLU()

        self.inplanes = 32
        self.layer1 = self._make_layer(32, 3, [2, 2])  # [32, 128]
        self.layer2 = self._make_layer(64, 4, [2, 2])  # [16, 64]
        self.layer3 = self._make_layer(128, 6, [2, 2])  # [8, 32]
        self.layer4 = self._make_layer(256, 6, [1, 1])  # [8, 32]
        self.layer5 = self._make_layer(512, 3, [1, 1])  # [8, 32]

        if self.compress_layer:
            self.layer6 = nn.Sequential(
                nn.Conv2D(
                    512, 256, kernel_size=(3, 1), padding=(0, 0), stride=(1,
                                                                          1)),
                nn.BatchNorm(256),
                nn.ReLU())
            self.out_channels = 256
        else:
            self.out_channels = 512

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            KaimingNormal(m.weight)
        elif isinstance(m, nn.BatchNorm):
            ones_(m.weight)
            zeros_(m.bias)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride), nn.BatchNorm2D(planes))

        layers = []
        layers.append(Block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(Block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_new(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        if not self.compress_layer:
            return x5
        else:
            x6 = self.layer6(x5)
            return x6


if __name__ == '__main__':
    model = ResNet45()
    x = paddle.rand([1, 3, 64, 256])
    x = paddle.to_tensor(x)
    print(x.shape)
    out = model(x)
    print(out.shape)
