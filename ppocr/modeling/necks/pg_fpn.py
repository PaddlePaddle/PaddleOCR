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


class PGFPN(nn.Layer):
    def __init__(self, in_channels, **kwargs):
        super(PGFPN, self).__init__()
        num_inputs = [2048, 2048, 1024, 512, 256]
        num_outputs = [256, 256, 192, 192, 128]
        self.out_channels = 128
        self.conv_bn_layer_1 = ConvBNLayer(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act=None,
            name='FPN_d1')
        self.conv_bn_layer_2 = ConvBNLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act=None,
            name='FPN_d2')
        self.conv_bn_layer_3 = ConvBNLayer(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            act=None,
            name='FPN_d3')
        self.conv_bn_layer_4 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            act=None,
            name='FPN_d4')
        self.conv_bn_layer_5 = ConvBNLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu',
            name='FPN_d5')
        self.conv_bn_layer_6 = ConvBNLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            act=None,
            name='FPN_d6')
        self.conv_bn_layer_7 = ConvBNLayer(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            act='relu',
            name='FPN_d7')
        self.conv_bn_layer_8 = ConvBNLayer(
            in_channels=128,
            out_channels=128,
            kernel_size=1,
            stride=1,
            act=None,
            name='FPN_d8')

        self.conv_h0 = ConvBNLayer(
            in_channels=num_inputs[0],
            out_channels=num_outputs[0],
            kernel_size=1,
            stride=1,
            act=None,
            name="conv_h{}".format(0))
        self.conv_h1 = ConvBNLayer(
            in_channels=num_inputs[1],
            out_channels=num_outputs[1],
            kernel_size=1,
            stride=1,
            act=None,
            name="conv_h{}".format(1))
        self.conv_h2 = ConvBNLayer(
            in_channels=num_inputs[2],
            out_channels=num_outputs[2],
            kernel_size=1,
            stride=1,
            act=None,
            name="conv_h{}".format(2))
        self.conv_h3 = ConvBNLayer(
            in_channels=num_inputs[3],
            out_channels=num_outputs[3],
            kernel_size=1,
            stride=1,
            act=None,
            name="conv_h{}".format(3))
        self.conv_h4 = ConvBNLayer(
            in_channels=num_inputs[4],
            out_channels=num_outputs[4],
            kernel_size=1,
            stride=1,
            act=None,
            name="conv_h{}".format(4))

        self.dconv0 = DeConvBNLayer(
            in_channels=num_outputs[0],
            out_channels=num_outputs[0 + 1],
            name="dconv_{}".format(0))
        self.dconv1 = DeConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[1 + 1],
            act=None,
            name="dconv_{}".format(1))
        self.dconv2 = DeConvBNLayer(
            in_channels=num_outputs[2],
            out_channels=num_outputs[2 + 1],
            act=None,
            name="dconv_{}".format(2))
        self.dconv3 = DeConvBNLayer(
            in_channels=num_outputs[3],
            out_channels=num_outputs[3 + 1],
            act=None,
            name="dconv_{}".format(3))
        self.conv_g1 = ConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[1],
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv_g{}".format(1))
        self.conv_g2 = ConvBNLayer(
            in_channels=num_outputs[2],
            out_channels=num_outputs[2],
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv_g{}".format(2))
        self.conv_g3 = ConvBNLayer(
            in_channels=num_outputs[3],
            out_channels=num_outputs[3],
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv_g{}".format(3))
        self.conv_g4 = ConvBNLayer(
            in_channels=num_outputs[4],
            out_channels=num_outputs[4],
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv_g{}".format(4))
        self.convf = ConvBNLayer(
            in_channels=num_outputs[4],
            out_channels=num_outputs[4],
            kernel_size=1,
            stride=1,
            act=None,
            name="conv_f{}".format(4))

    def forward(self, x):
        c0, c1, c2, c3, c4, c5, c6 = x
        # FPN_Down_Fusion
        f = [c0, c1, c2]
        g = [None, None, None]
        h = [None, None, None]
        h[0] = self.conv_bn_layer_1(f[0])
        h[1] = self.conv_bn_layer_2(f[1])
        h[2] = self.conv_bn_layer_3(f[2])

        g[0] = self.conv_bn_layer_4(h[0])
        g[1] = paddle.add(g[0], h[1])
        g[1] = F.relu(g[1])
        g[1] = self.conv_bn_layer_5(g[1])
        g[1] = self.conv_bn_layer_6(g[1])

        g[2] = paddle.add(g[1], h[2])
        g[2] = F.relu(g[2])
        g[2] = self.conv_bn_layer_7(g[2])
        f_down = self.conv_bn_layer_8(g[2])

        # FPN UP Fusion
        f1 = [c6, c5, c4, c3, c2]
        g = [None, None, None, None, None]
        h = [None, None, None, None, None]
        h[0] = self.conv_h0(f1[0])
        h[1] = self.conv_h1(f1[1])
        h[2] = self.conv_h2(f1[2])
        h[3] = self.conv_h3(f1[3])
        h[4] = self.conv_h4(f1[4])

        g[0] = self.dconv0(h[0])
        g[1] = paddle.add(g[0], h[1])
        g[1] = F.relu(g[1])
        g[1] = self.conv_g1(g[1])
        g[1] = self.dconv1(g[1])

        g[2] = paddle.add(g[1], h[2])
        g[2] = F.relu(g[2])
        g[2] = self.conv_g2(g[2])
        g[2] = self.dconv2(g[2])

        g[3] = paddle.add(g[2], h[3])
        g[3] = F.relu(g[3])
        g[3] = self.conv_g3(g[3])
        g[3] = self.dconv3(g[3])

        g[4] = paddle.add(x=g[3], y=h[4])
        g[4] = F.relu(g[4])
        g[4] = self.conv_g4(g[4])
        f_up = self.convf(g[4])
        f_common = paddle.add(f_down, f_up)
        f_common = F.relu(f_common)
        return f_common
