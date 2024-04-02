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

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.functional import hardswish, hardsigmoid
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.regularizer import L2Decay
import math

from paddle.utils.cpp_extension import load
# jit compile custom op
custom_ops = load(
    name="custom_jit_ops",
    sources=["./custom_op/custom_relu_op.cc", "./custom_op/custom_relu_op.cu"])


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3(nn.Layer):
    def __init__(self,
                 scale=1.0,
                 model_name="small",
                 dropout_prob=0.2,
                 class_dim=1000,
                 use_custom_relu=False):
        super(MobileNetV3, self).__init__()
        self.use_custom_relu = use_custom_relu

        inplanes = 16
        if model_name == "large":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, "relu", 1],
                [3, 64, 24, False, "relu", 2],
                [3, 72, 24, False, "relu", 1],
                [5, 72, 40, True, "relu", 2],
                [5, 120, 40, True, "relu", 1],
                [5, 120, 40, True, "relu", 1],
                [3, 240, 80, False, "hardswish", 2],
                [3, 200, 80, False, "hardswish", 1],
                [3, 184, 80, False, "hardswish", 1],
                [3, 184, 80, False, "hardswish", 1],
                [3, 480, 112, True, "hardswish", 1],
                [3, 672, 112, True, "hardswish", 1],
                [5, 672, 160, True, "hardswish", 2],
                [5, 960, 160, True, "hardswish", 1],
                [5, 960, 160, True, "hardswish", 1],
            ]
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        elif model_name == "small":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, "relu", 2],
                [3, 72, 24, False, "relu", 2],
                [3, 88, 24, False, "relu", 1],
                [5, 96, 40, True, "hardswish", 2],
                [5, 240, 40, True, "hardswish", 1],
                [5, 240, 40, True, "hardswish", 1],
                [5, 120, 48, True, "hardswish", 1],
                [5, 144, 48, True, "hardswish", 1],
                [5, 288, 96, True, "hardswish", 2],
                [5, 576, 96, True, "hardswish", 1],
                [5, 576, 96, True, "hardswish", 1],
            ]
            self.cls_ch_squeeze = 576
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError(
                "mode[{}_model] is not implemented!".format(model_name))

        self.conv1 = ConvBNLayer(
            in_c=3,
            out_c=make_divisible(inplanes * scale),
            filter_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act="hardswish",
            name="conv1",
            use_custom_relu=self.use_custom_relu)

        self.block_list = []
        i = 0
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in self.cfg:
            block = self.add_sublayer(
                "conv" + str(i + 2),
                ResidualUnit(
                    in_c=inplanes,
                    mid_c=make_divisible(scale * exp),
                    out_c=make_divisible(scale * c),
                    filter_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                    name="conv" + str(i + 2),
                    use_custom_relu=self.use_custom_relu))
            self.block_list.append(block)
            inplanes = make_divisible(scale * c)
            i += 1

        self.last_second_conv = ConvBNLayer(
            in_c=inplanes,
            out_c=make_divisible(scale * self.cls_ch_squeeze),
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act="hardswish",
            name="conv_last",
            use_custom_relu=self.use_custom_relu)

        self.pool = AdaptiveAvgPool2D(1)

        self.last_conv = Conv2D(
            in_channels=make_divisible(scale * self.cls_ch_squeeze),
            out_channels=self.cls_ch_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(),
            bias_attr=False)

        self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")

        self.out = Linear(
            self.cls_ch_expand,
            class_dim,
            weight_attr=ParamAttr(),
            bias_attr=ParamAttr())

    def forward(self, inputs, label=None):
        x = self.conv1(inputs)

        for block in self.block_list:
            x = block(x)

        x = self.last_second_conv(x)
        x = self.pool(x)

        x = self.last_conv(x)
        x = hardswish(x)
        x = self.dropout(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.out(x)
        return x


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 if_act=True,
                 act=None,
                 use_cudnn=True,
                 name="",
                 use_custom_relu=False):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = Conv2D(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(),
            bias_attr=False)
        self.bn = BatchNorm(
            num_channels=out_c,
            act=None,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        # moving_mean_name=name + "_bn_mean",
        # moving_variance_name=name + "_bn_variance")

        self.use_custom_relu = use_custom_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                if self.use_custom_relu:
                    x = custom_ops.custom_relu(x)
                else:
                    x = F.relu(x)
            elif self.act == "hardswish":
                x = hardswish(x)
            else:
                print("The activation function is selected incorrectly.")
                exit()
        return x


class ResidualUnit(nn.Layer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 act=None,
                 name='',
                 use_custom_relu=False):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se

        self.use_custom_relu = use_custom_relu

        self.expand_conv = ConvBNLayer(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + "_expand",
            use_custom_relu=self.use_custom_relu)
        self.bottleneck_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            num_groups=mid_c,
            if_act=True,
            act=act,
            name=name + "_depthwise",
            use_custom_relu=self.use_custom_relu)
        if self.if_se:
            self.mid_se = SEModule(mid_c, name=name + "_se")
        self.linear_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name=name + "_linear",
            use_custom_relu=self.use_custom_relu)

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
    def __init__(self, channel, reduction=4, name=""):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(),
            bias_attr=ParamAttr())
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(),
            bias_attr=ParamAttr())

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = hardsigmoid(outputs, slope=0.2, offset=0.5)
        return paddle.multiply(x=inputs, y=outputs)


def MobileNetV3_small_x0_35(**args):
    model = MobileNetV3(model_name="small", scale=0.35, **args)
    return model


def MobileNetV3_small_x0_5(**args):
    model = MobileNetV3(model_name="small", scale=0.5, **args)
    return model


def MobileNetV3_small_x0_75(**args):
    model = MobileNetV3(model_name="small", scale=0.75, **args)
    return model


def MobileNetV3_small_x1_0(**args):
    model = MobileNetV3(model_name="small", scale=1.0, **args)
    return model


def MobileNetV3_small_x1_25(**args):
    model = MobileNetV3(model_name="small", scale=1.25, **args)
    return model


def MobileNetV3_large_x0_35(**args):
    model = MobileNetV3(model_name="large", scale=0.35, **args)
    return model


def MobileNetV3_large_x0_5(**args):
    model = MobileNetV3(model_name="large", scale=0.5, **args)
    return model


def MobileNetV3_large_x0_75(**args):
    model = MobileNetV3(model_name="large", scale=0.75, **args)
    return model


def MobileNetV3_large_x1_0(**args):
    model = MobileNetV3(model_name="large", scale=1.0, **args)
    return model


def MobileNetV3_large_x1_25(**args):
    model = MobileNetV3(model_name="large", scale=1.25, **args)
    return


class DistillMV3(nn.Layer):
    def __init__(self,
                 scale=1.0,
                 model_name="small",
                 dropout_prob=0.2,
                 class_dim=1000,
                 args=None,
                 use_custom_relu=False):
        super(DistillMV3, self).__init__()

        self.student = MobileNetV3(
            model_name=model_name,
            scale=scale,
            class_dim=class_dim,
            use_custom_relu=use_custom_relu)

        self.student1 = MobileNetV3(
            model_name=model_name,
            scale=scale,
            class_dim=class_dim,
            use_custom_relu=use_custom_relu)

    def forward(self, inputs, label=None):
        predicts = dict()
        predicts['student'] = self.student(inputs, label)
        predicts['student1'] = self.student1(inputs, label)
        return predicts


def distillmv3_large_x0_5(**args):
    model = DistillMV3(model_name="large", scale=0.5, **args)
    return model


class SiameseMV3(nn.Layer):
    def __init__(self,
                 scale=1.0,
                 model_name="small",
                 dropout_prob=0.2,
                 class_dim=1000,
                 args=None,
                 use_custom_relu=False):
        super(SiameseMV3, self).__init__()

        self.net = MobileNetV3(
            model_name=model_name,
            scale=scale,
            class_dim=class_dim,
            use_custom_relu=use_custom_relu)
        self.net1 = MobileNetV3(
            model_name=model_name,
            scale=scale,
            class_dim=class_dim,
            use_custom_relu=use_custom_relu)

    def forward(self, inputs, label=None):
        # net
        x = self.net.conv1(inputs)
        for block in self.net.block_list:
            x = block(x)

        # net1 
        x1 = self.net1.conv1(inputs)
        for block in self.net1.block_list:
            x1 = block(x1)
        # add
        x = x + x1

        x = self.net.last_second_conv(x)
        x = self.net.pool(x)

        x = self.net.last_conv(x)
        x = hardswish(x)
        x = self.net.dropout(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.net.out(x)
        return x


def siamese_mv3(class_dim, use_custom_relu):
    model = SiameseMV3(
        scale=0.5,
        model_name="large",
        class_dim=class_dim,
        use_custom_relu=use_custom_relu)
    return model


def build_model(config):
    model_type = config['model_type']
    if model_type == "cls":
        class_dim = config['MODEL']['class_dim']
        use_custom_relu = config['MODEL']['use_custom_relu']
        if 'siamese' in config['MODEL'] and config['MODEL']['siamese'] is True:
            model = siamese_mv3(
                class_dim=class_dim, use_custom_relu=use_custom_relu)
        else:
            model = MobileNetV3_large_x0_5(
                class_dim=class_dim, use_custom_relu=use_custom_relu)

    elif model_type == "cls_distill":
        class_dim = config['MODEL']['class_dim']
        use_custom_relu = config['MODEL']['use_custom_relu']
        model = distillmv3_large_x0_5(
            class_dim=class_dim, use_custom_relu=use_custom_relu)

    elif model_type == "cls_distill_multiopt":
        class_dim = config['MODEL']['class_dim']
        use_custom_relu = config['MODEL']['use_custom_relu']
        model = distillmv3_large_x0_5(
            class_dim=100, use_custom_relu=use_custom_relu)
    else:
        raise ValueError("model_type should be one of ['']")

    return model
