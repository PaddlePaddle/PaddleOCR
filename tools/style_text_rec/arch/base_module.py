#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import paddle
import paddle.nn as nn
from arch.spectral_norm import spectral_norm


class CBN(nn.Layer):
    def __init__(self,
                 name,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 norm_layer=None,
                 act=None,
                 act_attr=None):
        super(CBN, self).__init__()
        if use_bias:
            bias_attr = paddle.ParamAttr(name=name + "_bias")
        else:
            bias_attr = None
        self._conv = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            weight_attr=paddle.ParamAttr(name=name + "_weights"),
            bias_attr=bias_attr)
        if norm_layer:
            self._norm_layer = getattr(paddle.nn, norm_layer)(
                num_features=out_channels, name=name + "_bn")
        else:
            self._norm_layer = None
        if act:
            if act_attr:
                self._act = getattr(paddle.nn, act)(**act_attr,
                                                    name=name + "_" + act)
            else:
                self._act = getattr(paddle.nn, act)(name=name + "_" + act)
        else:
            self._act = None

    def forward(self, x):
        out = self._conv(x)
        if self._norm_layer:
            out = self._norm_layer(out)
        if self._act:
            out = self._act(out)
        return out


class SNConv(nn.Layer):
    def __init__(self,
                 name,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 norm_layer=None,
                 act=None,
                 act_attr=None):
        super(SNConv, self).__init__()
        if use_bias:
            bias_attr = paddle.ParamAttr(name=name + "_bias")
        else:
            bias_attr = None
        self._sn_conv = spectral_norm(
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                weight_attr=paddle.ParamAttr(name=name + "_weights"),
                bias_attr=bias_attr))
        if norm_layer:
            self._norm_layer = getattr(paddle.nn, norm_layer)(
                num_features=out_channels, name=name + "_bn")
        else:
            self._norm_layer = None
        if act:
            if act_attr:
                self._act = getattr(paddle.nn, act)(**act_attr,
                                                    name=name + "_" + act)
            else:
                self._act = getattr(paddle.nn, act)(name=name + "_" + act)
        else:
            self._act = None

    def forward(self, x):
        out = self._sn_conv(x)
        if self._norm_layer:
            out = self._norm_layer(out)
        if self._act:
            out = self._act(out)
        return out


class SNConvTranspose(nn.Layer):
    def __init__(self,
                 name,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 norm_layer=None,
                 act=None,
                 act_attr=None):
        super(SNConvTranspose, self).__init__()
        if use_bias:
            bias_attr = paddle.ParamAttr(name=name + "_bias")
        else:
            bias_attr = None
        self._sn_conv_transpose = spectral_norm(
            paddle.nn.Conv2DTranspose(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                groups=groups,
                weight_attr=paddle.ParamAttr(name=name + "_weights"),
                bias_attr=bias_attr))
        if norm_layer:
            self._norm_layer = getattr(paddle.nn, norm_layer)(
                num_features=out_channels, name=name + "_bn")
        else:
            self._norm_layer = None
        if act:
            if act_attr:
                self._act = getattr(paddle.nn, act)(**act_attr,
                                                    name=name + "_" + act)
            else:
                self._act = getattr(paddle.nn, act)(name=name + "_" + act)
        else:
            self._act = None

    def forward(self, x):
        out = self._sn_conv_transpose(x)
        if self._norm_layer:
            out = self._norm_layer(out)
        if self._act:
            out = self._act(out)
        return out


class MiddleNet(nn.Layer):
    def __init__(self, name, in_channels, mid_channels, out_channels,
                 use_bias):
        super(MiddleNet, self).__init__()
        self._sn_conv1 = SNConv(
            name=name + "_sn_conv1",
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            use_bias=use_bias,
            norm_layer=None,
            act=None)
        self._pad2d = nn.Pad2D(padding=[1, 1, 1, 1], mode="replicate")
        self._sn_conv2 = SNConv(
            name=name + "_sn_conv2",
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            use_bias=use_bias)
        self._sn_conv3 = SNConv(
            name=name + "_sn_conv3",
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_bias=use_bias)

    def forward(self, x):

        sn_conv1 = self._sn_conv1.forward(x)
        pad_2d = self._pad2d.forward(sn_conv1)
        sn_conv2 = self._sn_conv2.forward(pad_2d)
        sn_conv3 = self._sn_conv3.forward(sn_conv2)
        return sn_conv3


class ResBlock(nn.Layer):
    def __init__(self, name, channels, norm_layer, use_dropout, use_dilation,
                 use_bias):
        super(ResBlock, self).__init__()
        if use_dilation:
            padding_mat = [1, 1, 1, 1]
        else:
            padding_mat = [0, 0, 0, 0]
        self._pad1 = nn.Pad2D(padding_mat, mode="replicate")

        self._sn_conv1 = SNConv(
            name=name + "_sn_conv1",
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=0,
            norm_layer=norm_layer,
            use_bias=use_bias,
            act="ReLU",
            act_attr=None)
        if use_dropout:
            self._dropout = nn.Dropout(0.5)
        else:
            self._dropout = None
        self._pad2 = nn.Pad2D([1, 1, 1, 1], mode="replicate")
        self._sn_conv2 = SNConv(
            name=name + "_sn_conv2",
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            norm_layer=norm_layer,
            use_bias=use_bias,
            act="ReLU",
            act_attr=None)

    def forward(self, x):
        pad1 = self._pad1.forward(x)
        sn_conv1 = self._sn_conv1.forward(pad1)
        pad2 = self._pad2.forward(sn_conv1)
        sn_conv2 = self._sn_conv2.forward(pad2)
        return sn_conv2 + x
