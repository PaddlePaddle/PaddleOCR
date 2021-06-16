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


class TableFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(TableFPN, self).__init__()
        self.out_channels = 512
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.in2_conv = nn.Conv2D(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(
                name='conv2d_51.w_0', initializer=weight_attr),
            bias_attr=False)
        self.in3_conv = nn.Conv2D(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            stride = 1,
            weight_attr=ParamAttr(
                name='conv2d_50.w_0', initializer=weight_attr),
            bias_attr=False)
        self.in4_conv = nn.Conv2D(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(
                name='conv2d_49.w_0', initializer=weight_attr),
            bias_attr=False)
        self.in5_conv = nn.Conv2D(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(
                name='conv2d_48.w_0', initializer=weight_attr),
            bias_attr=False)
        self.p5_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(
                name='conv2d_52.w_0', initializer=weight_attr),
            bias_attr=False)
        self.p4_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(
                name='conv2d_53.w_0', initializer=weight_attr),
            bias_attr=False)
        self.p3_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(
                name='conv2d_54.w_0', initializer=weight_attr),
            bias_attr=False)
        self.p2_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(
                name='conv2d_55.w_0', initializer=weight_attr),
            bias_attr=False)
        self.fuse_conv = nn.Conv2D(
            in_channels=self.out_channels * 4,
            out_channels=512,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(
                name='conv2d_fuse.w_0', initializer=weight_attr), bias_attr=False)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.upsample(
            in5, size=in4.shape[2:4], mode="nearest", align_mode=1)  # 1/16
        out3 = in3 + F.upsample(
            out4, size=in3.shape[2:4], mode="nearest", align_mode=1)  # 1/8
        out2 = in2 + F.upsample(
            out3, size=in2.shape[2:4], mode="nearest", align_mode=1)  # 1/4

        p4 = F.upsample(out4, size=in5.shape[2:4], mode="nearest", align_mode=1)
        p3 = F.upsample(out3, size=in5.shape[2:4], mode="nearest", align_mode=1)
        p2 = F.upsample(out2, size=in5.shape[2:4], mode="nearest", align_mode=1)
        fuse = paddle.concat([in5, p4, p3, p2], axis=1)
        fuse_conv = self.fuse_conv(fuse) * 0.005
        return [c5 + fuse_conv]
