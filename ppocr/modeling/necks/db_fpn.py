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


class DBFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        weight_attr = paddle.nn.initializer.MSRA(uniform=False)

        self.in2_conv = nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(
                name='conv2d_51.w_0', initializer=weight_attr),
            bias_attr=False)
        self.in3_conv = nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(
                name='conv2d_50.w_0', initializer=weight_attr),
            bias_attr=False)
        self.in4_conv = nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(
                name='conv2d_49.w_0', initializer=weight_attr),
            bias_attr=False)
        self.in5_conv = nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(
                name='conv2d_48.w_0', initializer=weight_attr),
            bias_attr=False)
        self.p5_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(
                name='conv2d_52.w_0', initializer=weight_attr),
            bias_attr=False)
        self.p4_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(
                name='conv2d_53.w_0', initializer=weight_attr),
            bias_attr=False)
        self.p3_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(
                name='conv2d_54.w_0', initializer=weight_attr),
            bias_attr=False)
        self.p2_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(
                name='conv2d_55.w_0', initializer=weight_attr),
            bias_attr=False)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.resize_nearest(in5, scale=2)  # 1/16
        out3 = in3 + F.resize_nearest(out4, scale=2)  # 1/8
        out2 = in2 + F.resize_nearest(out3, scale=2)  # 1/4

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)
        p5 = F.resize_nearest(p5, scale=8)
        p4 = F.resize_nearest(p4, scale=4)
        p3 = F.resize_nearest(p3, scale=2)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        return fuse
