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

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
import os
import sys

import math
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
ones_ = Constant(value=1.)
zeros_ = Constant(value=0.)

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))


class Conv_BN_ReLU(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=False)
        self.bn = nn.BatchNorm2D(out_planes)
        self.relu = nn.ReLU()

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                normal_ = Normal(mean=0.0, std=math.sqrt(2. / n))
                normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                zeros_(m.bias)
                ones_(m.weight)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPEM(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(FPEM, self).__init__()
        planes = out_channels
        self.dwconv3_1 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=planes,
            bias_attr=False)
        self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_1 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=planes,
            bias_attr=False)
        self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv1_1 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=planes,
            bias_attr=False)
        self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_2 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=planes,
            bias_attr=False)
        self.smooth_layer2_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv3_2 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=planes,
            bias_attr=False)
        self.smooth_layer3_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv4_2 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=planes,
            bias_attr=False)
        self.smooth_layer4_2 = Conv_BN_ReLU(planes, planes)

    def _upsample_add(self, x, y):
        return F.upsample(x, scale_factor=2, mode='bilinear') + y

    def forward(self, f1, f2, f3, f4):
        # up-down
        f3 = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        f2 = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3, f2)))
        f1 = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2, f1)))

        # down-up
        f2 = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2, f1)))
        f3 = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3, f2)))
        f4 = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3)))

        return f1, f2, f3, f4


class CTFPN(nn.Layer):
    def __init__(self, in_channels, out_channel=128):
        super(CTFPN, self).__init__()
        self.out_channels = out_channel * 4

        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)

        self.fpem1 = FPEM(in_channels=(64, 128, 256, 512), out_channels=128)
        self.fpem2 = FPEM(in_channels=(64, 128, 256, 512), out_channels=128)

    def _upsample(self, x, scale=1):
        return F.upsample(x, scale_factor=scale, mode='bilinear')

    def forward(self, f):
        # # reduce channel
        f1 = self.reduce_layer1(f[0])  # N,64,160,160    --> N, 128, 160, 160
        f2 = self.reduce_layer2(f[1])  # N, 128, 80, 80  --> N, 128, 80, 80
        f3 = self.reduce_layer3(f[2])  # N, 256, 40, 40  --> N, 128, 40, 40
        f4 = self.reduce_layer4(f[3])  # N, 512, 20, 20  --> N, 128, 20, 20

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2

        f2 = self._upsample(f2, scale=2)
        f3 = self._upsample(f3, scale=4)
        f4 = self._upsample(f4, scale=8)
        ff = paddle.concat((f1, f2, f3, f4), 1)  # N,512, 160,160
        return ff
