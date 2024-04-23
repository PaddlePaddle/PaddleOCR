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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import nn, ParamAttr
from paddle.nn import functional as F
import paddle
import numpy as np

__all__ = ["ResNetFPN"]


class ResNetFPN(nn.Layer):
    def __init__(self, in_channels=1, layers=50, **kwargs):
        super(ResNetFPN, self).__init__()
        supported_layers = {
            18: {"depth": [2, 2, 2, 2], "block_class": BasicBlock},
            34: {"depth": [3, 4, 6, 3], "block_class": BasicBlock},
            50: {"depth": [3, 4, 6, 3], "block_class": BottleneckBlock},
            101: {"depth": [3, 4, 23, 3], "block_class": BottleneckBlock},
            152: {"depth": [3, 8, 36, 3], "block_class": BottleneckBlock},
        }
        stride_list = [(2, 2), (2, 2), (1, 1), (1, 1)]
        num_filters = [64, 128, 256, 512]
        self.depth = supported_layers[layers]["depth"]
        self.F = []
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            act="relu",
            name="conv1",
        )
        self.block_list = []
        in_ch = 64
        if layers >= 50:
            for block in range(len(self.depth)):
                for i in range(self.depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    block_list = self.add_sublayer(
                        "bottleneckBlock_{}_{}".format(block, i),
                        BottleneckBlock(
                            in_channels=in_ch,
                            out_channels=num_filters[block],
                            stride=stride_list[block] if i == 0 else 1,
                            name=conv_name,
                        ),
                    )
                    in_ch = num_filters[block] * 4
                    self.block_list.append(block_list)
                self.F.append(block_list)
        else:
            for block in range(len(self.depth)):
                for i in range(self.depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    if i == 0 and block != 0:
                        stride = (2, 1)
                    else:
                        stride = (1, 1)
                    basic_block = self.add_sublayer(
                        conv_name,
                        BasicBlock(
                            in_channels=in_ch,
                            out_channels=num_filters[block],
                            stride=stride_list[block] if i == 0 else 1,
                            is_first=block == i == 0,
                            name=conv_name,
                        ),
                    )
                    in_ch = basic_block.out_channels
                    self.block_list.append(basic_block)
        out_ch_list = [in_ch // 4, in_ch // 2, in_ch]
        self.base_block = []
        self.conv_trans = []
        self.bn_block = []
        for i in [-2, -3]:
            in_channels = out_ch_list[i + 1] + out_ch_list[i]

            self.base_block.append(
                self.add_sublayer(
                    "F_{}_base_block_0".format(i),
                    nn.Conv2D(
                        in_channels=in_channels,
                        out_channels=out_ch_list[i],
                        kernel_size=1,
                        weight_attr=ParamAttr(trainable=True),
                        bias_attr=ParamAttr(trainable=True),
                    ),
                )
            )
            self.base_block.append(
                self.add_sublayer(
                    "F_{}_base_block_1".format(i),
                    nn.Conv2D(
                        in_channels=out_ch_list[i],
                        out_channels=out_ch_list[i],
                        kernel_size=3,
                        padding=1,
                        weight_attr=ParamAttr(trainable=True),
                        bias_attr=ParamAttr(trainable=True),
                    ),
                )
            )
            self.base_block.append(
                self.add_sublayer(
                    "F_{}_base_block_2".format(i),
                    nn.BatchNorm(
                        num_channels=out_ch_list[i],
                        act="relu",
                        param_attr=ParamAttr(trainable=True),
                        bias_attr=ParamAttr(trainable=True),
                    ),
                )
            )
        self.base_block.append(
            self.add_sublayer(
                "F_{}_base_block_3".format(i),
                nn.Conv2D(
                    in_channels=out_ch_list[i],
                    out_channels=512,
                    kernel_size=1,
                    bias_attr=ParamAttr(trainable=True),
                    weight_attr=ParamAttr(trainable=True),
                ),
            )
        )
        self.out_channels = 512

    def __call__(self, x):
        x = self.conv(x)
        fpn_list = []
        F = []
        for i in range(len(self.depth)):
            fpn_list.append(np.sum(self.depth[: i + 1]))

        for i, block in enumerate(self.block_list):
            x = block(x)
            for number in fpn_list:
                if i + 1 == number:
                    F.append(x)
        base = F[-1]

        j = 0
        for i, block in enumerate(self.base_block):
            if i % 3 == 0 and i < 6:
                j = j + 1
                b, c, w, h = F[-j - 1].shape
                if [w, h] == list(base.shape[2:]):
                    base = base
                else:
                    base = self.conv_trans[j - 1](base)
                    base = self.bn_block[j - 1](base)
                base = paddle.concat([base, F[-j - 1]], axis=1)
            base = block(base)
        return base


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        act=None,
        name=None,
    ):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 if stride == (1, 1) else kernel_size,
            dilation=2 if stride == (1, 1) else 1,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + ".conv2d.output.1.w_0"),
            bias_attr=False,
        )

        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name=name + ".output.1.w_0"),
            bias_attr=ParamAttr(name=name + ".output.1.b_0"),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance",
        )

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ShortCut(nn.Layer):
    def __init__(self, in_channels, out_channels, stride, name, is_first=False):
        super(ShortCut, self).__init__()
        self.use_conv = True

        if in_channels != out_channels or stride != 1 or is_first == True:
            if stride == (1, 1):
                self.conv = ConvBNLayer(in_channels, out_channels, 1, 1, name=name)
            else:  # stride==(2,2)
                self.conv = ConvBNLayer(in_channels, out_channels, 1, stride, name=name)
        else:
            self.use_conv = False

    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)
        return x


class BottleneckBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride, name):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act="relu",
            name=name + "_branch2a",
        )
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act="relu",
            name=name + "_branch2b",
        )

        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + "_branch2c",
        )

        self.short = ShortCut(
            in_channels=in_channels,
            out_channels=out_channels * 4,
            stride=stride,
            is_first=False,
            name=name + "_branch1",
        )
        self.out_channels = out_channels * 4

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = y + self.short(x)
        y = F.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride, name, is_first):
        super(BasicBlock, self).__init__()
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            act="relu",
            stride=stride,
            name=name + "_branch2a",
        )
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None,
            name=name + "_branch2b",
        )
        self.short = ShortCut(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            is_first=is_first,
            name=name + "_branch1",
        )
        self.out_channels = out_channels

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = y + self.short(x)
        return F.relu(y)
