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

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr

__all__ = ['MobileNetV3']


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 model_name='large',
                 scale=0.5,
                 disable_se=False,
                 **kwargs):
        """
        the MobilenetV3 backbone network for detection module.
        Args:
            params(dict): the super parameters for build network
        """
        super(MobileNetV3, self).__init__()

        self.disable_se = disable_se

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hardswish', 2],
                [3, 200, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 480, 112, True, 'hardswish', 1],
                [3, 672, 112, True, 'hardswish', 1],
                [5, 672, 160, True, 'hardswish', 2],
                [5, 960, 160, True, 'hardswish', 1],
                [5, 960, 160, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hardswish', 2],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 120, 48, True, 'hardswish', 1],
                [5, 144, 48, True, 'hardswish', 1],
                [5, 288, 96, True, 'hardswish', 2],
                [5, 576, 96, True, 'hardswish', 1],
                [5, 576, 96, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, scale)
        inplanes = 16
        # conv1
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv1')

        self.stages = []
        self.out_channels = []
        block_list = []
        i = 0
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in cfg:
            se = se and not self.disable_se
            if s == 2 and i > 2:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                    name="conv" + str(i + 2)))
            inplanes = make_divisible(scale * c)
            i += 1
        block_list.append(
            ConvBNLayer(
                in_channels=inplanes,
                out_channels=make_divisible(scale * cls_ch_squeeze),
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                if_act=True,
                act='hardswish',
                name='conv_last'))
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))
        for i, stage in enumerate(self.stages):
            self.add_sublayer(sublayer=stage, name="stage{}".format(i))

    def forward(self, x):
        x = self.conv(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2D(
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
            act=None,
            param_attr=ParamAttr(name=name + "_bn_scale"),
            bias_attr=ParamAttr(name=name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()
        return x


class ResidualUnit(nn.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 use_se,
                 act=None,
                 name=''):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + "_expand")
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2),
            groups=mid_channels,
            if_act=True,
            act=act,
            name=name + "_depthwise")
        if self.if_se:
            self.mid_se = SEModule(mid_channels, name=name + "_se")
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name=name + "_linear")

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.add(inputs, x)
        return x


class SEModule(nn.Layer):
    def __init__(self, in_channels, reduction=4, name=""):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name=name + "_1_weights"),
            bias_attr=ParamAttr(name=name + "_1_offset"))
        self.conv2 = nn.Conv2D(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name + "_2_weights"),
            bias_attr=ParamAttr(name=name + "_2_offset"))

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs, slope=0.2, offset=0.5)
        return inputs * outputs
