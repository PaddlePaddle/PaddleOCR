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


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 is_vd_mode=False,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self._batch_norm = nn.BatchNorm(
            out_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
            use_global_stats=False)

    def forward(self, inputs):
        # if self.is_vd_mode:
        #     inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class DeConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(DeConvBNLayer, self).__init__()

        self.if_act = if_act
        self.act = act
        self.deconv = nn.Conv2DTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)
        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name="bn_" + name + "_scale"),
            bias_attr=ParamAttr(name="bn_" + name + "_offset"),
            moving_mean_name="bn_" + name + "_mean",
            moving_variance_name="bn_" + name + "_variance",
            use_global_stats=False)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        return x


class FPN_Up_Fusion(nn.Layer):
    def __init__(self, in_channels):
        super(FPN_Up_Fusion, self).__init__()
        in_channels = in_channels[::-1]
        out_channels = [256, 256, 192, 192, 128]

        self.h0_conv = ConvBNLayer(
            in_channels[0], out_channels[0], 1, 1, act=None, name='conv_h0')
        self.h1_conv = ConvBNLayer(
            in_channels[1], out_channels[1], 1, 1, act=None, name='conv_h1')
        self.h2_conv = ConvBNLayer(
            in_channels[2], out_channels[2], 1, 1, act=None, name='conv_h2')
        self.h3_conv = ConvBNLayer(
            in_channels[3], out_channels[3], 1, 1, act=None, name='conv_h3')
        self.h4_conv = ConvBNLayer(
            in_channels[4], out_channels[4], 1, 1, act=None, name='conv_h4')

        self.dconv0 = DeConvBNLayer(
            in_channels=out_channels[0],
            out_channels=out_channels[1],
            name="dconv_{}".format(0))
        self.dconv1 = DeConvBNLayer(
            in_channels=out_channels[1],
            out_channels=out_channels[2],
            act=None,
            name="dconv_{}".format(1))
        self.dconv2 = DeConvBNLayer(
            in_channels=out_channels[2],
            out_channels=out_channels[3],
            act=None,
            name="dconv_{}".format(2))
        self.dconv3 = DeConvBNLayer(
            in_channels=out_channels[3],
            out_channels=out_channels[4],
            act=None,
            name="dconv_{}".format(3))
        self.conv_g1 = ConvBNLayer(
            in_channels=out_channels[1],
            out_channels=out_channels[1],
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv_g{}".format(1))
        self.conv_g2 = ConvBNLayer(
            in_channels=out_channels[2],
            out_channels=out_channels[2],
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv_g{}".format(2))
        self.conv_g3 = ConvBNLayer(
            in_channels=out_channels[3],
            out_channels=out_channels[3],
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv_g{}".format(3))
        self.conv_g4 = ConvBNLayer(
            in_channels=out_channels[4],
            out_channels=out_channels[4],
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv_g{}".format(4))
        self.convf = ConvBNLayer(
            in_channels=out_channels[4],
            out_channels=out_channels[4],
            kernel_size=1,
            stride=1,
            act=None,
            name="conv_f{}".format(4))

    def _add_relu(self, x1, x2):
        x = paddle.add(x=x1, y=x2)
        x = F.relu(x)
        return x

    def forward(self, x):
        f = x[2:][::-1]
        h0 = self.h0_conv(f[0])
        h1 = self.h1_conv(f[1])
        h2 = self.h2_conv(f[2])
        h3 = self.h3_conv(f[3])
        h4 = self.h4_conv(f[4])

        g0 = self.dconv0(h0)

        g1 = self.dconv2(self.conv_g2(self._add_relu(g0, h1)))
        g2 = self.dconv2(self.conv_g2(self._add_relu(g1, h2)))
        g3 = self.dconv3(self.conv_g2(self._add_relu(g2, h3)))
        g4 = self.dconv4(self.conv_g2(self._add_relu(g3, h4)))
        return g4


class FPN_Down_Fusion(nn.Layer):
    def __init__(self, in_channels):
        super(FPN_Down_Fusion, self).__init__()
        out_channels = [32, 64, 128]

        self.h0_conv = ConvBNLayer(
            in_channels[0], out_channels[0], 3, 1, act=None, name='FPN_d1')
        self.h1_conv = ConvBNLayer(
            in_channels[1], out_channels[1], 3, 1, act=None, name='FPN_d2')
        self.h2_conv = ConvBNLayer(
            in_channels[2], out_channels[2], 3, 1, act=None, name='FPN_d3')

        self.g0_conv = ConvBNLayer(
            out_channels[0], out_channels[1], 3, 2, act=None, name='FPN_d4')

        self.g1_conv = nn.Sequential(
            ConvBNLayer(
                out_channels[1],
                out_channels[1],
                3,
                1,
                act='relu',
                name='FPN_d5'),
            ConvBNLayer(
                out_channels[1], out_channels[2], 3, 2, act=None,
                name='FPN_d6'))

        self.g2_conv = nn.Sequential(
            ConvBNLayer(
                out_channels[2],
                out_channels[2],
                3,
                1,
                act='relu',
                name='FPN_d7'),
            ConvBNLayer(
                out_channels[2], out_channels[2], 1, 1, act=None,
                name='FPN_d8'))

    def forward(self, x):
        f = x[:3]
        h0 = self.h0_conv(f[0])
        h1 = self.h1_conv(f[1])
        h2 = self.h2_conv(f[2])
        g0 = self.g0_conv(h0)
        g1 = paddle.add(x=g0, y=h1)
        g1 = F.relu(g1)
        g1 = self.g1_conv(g1)
        g2 = paddle.add(x=g1, y=h2)
        g2 = F.relu(g2)
        g2 = self.g2_conv(g2)
        return g2


class PGFPN(nn.Layer):
    def __init__(self, in_channels, with_cab=False, **kwargs):
        super(PGFPN, self).__init__()
        self.in_channels = in_channels
        self.with_cab = with_cab
        self.FPN_Down_Fusion = FPN_Down_Fusion(self.in_channels)
        self.FPN_Up_Fusion = FPN_Up_Fusion(self.in_channels)
        self.out_channels = 128

    def forward(self, x):
        # down fpn
        f_down = self.FPN_Down_Fusion(x)

        # up fpn
        f_up = self.FPN_Up_Fusion(x)

        # fusion
        f_common = paddle.add(x=f_down, y=f_up)
        f_common = F.relu(f_common)

        return f_common
