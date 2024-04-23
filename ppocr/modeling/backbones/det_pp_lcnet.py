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

from __future__ import absolute_import, division, print_function

import os
import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, BatchNorm, Conv2D, Dropout, Linear
from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingNormal
from paddle.utils.download import get_path_from_url

MODEL_URLS = {
    "PPLCNet_x0.25": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams",
    "PPLCNet_x0.35": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_35_pretrained.pdparams",
    "PPLCNet_x0.5": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_pretrained.pdparams",
    "PPLCNet_x0.75": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_75_pretrained.pdparams",
    "PPLCNet_x1.0": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_pretrained.pdparams",
    "PPLCNet_x1.5": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_5_pretrained.pdparams",
    "PPLCNet_x2.0": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_0_pretrained.pdparams",
    "PPLCNet_x2.5": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_pretrained.pdparams",
}

MODEL_STAGES_PATTERN = {
    "PPLCNet": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
}

__all__ = list(MODEL_URLS.keys())

# Each element(list) represents a depthwise block, which is composed of k, in_c, out_c, s, use_se.
# k: kernel_size
# in_c: input channel number in depthwise block
# out_c: output channel number in depthwise block
# s: stride in depthwise block
# use_se: whether to use SE block

NET_CONFIG = {
    "blocks2":
    # k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [
        [3, 128, 256, 2, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
    ],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]],
}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Layer):
    def __init__(self, num_channels, filter_size, num_filters, stride, num_groups=1):
        super().__init__()

        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False,
        )

        self.bn = BatchNorm(
            num_filters,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)),
        )
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x


class DepthwiseSeparable(nn.Layer):
    def __init__(self, num_channels, num_filters, stride, dw_size=3, use_se=False):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels,
        )
        if use_se:
            self.se = SEModule(num_channels)
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels, filter_size=1, num_filters=num_filters, stride=1
        )

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.relu = nn.ReLU()
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class PPLCNet(nn.Layer):
    def __init__(self, in_channels=3, scale=1.0, pretrained=False, use_ssld=False):
        super().__init__()
        self.out_channels = [
            int(NET_CONFIG["blocks3"][-1][2] * scale),
            int(NET_CONFIG["blocks4"][-1][2] * scale),
            int(NET_CONFIG["blocks5"][-1][2] * scale),
            int(NET_CONFIG["blocks6"][-1][2] * scale),
        ]
        self.scale = scale

        self.conv1 = ConvBNLayer(
            num_channels=in_channels,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=2,
        )

        self.blocks2 = nn.Sequential(
            *[
                DepthwiseSeparable(
                    num_channels=make_divisible(in_c * scale),
                    num_filters=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks2"])
            ]
        )

        self.blocks3 = nn.Sequential(
            *[
                DepthwiseSeparable(
                    num_channels=make_divisible(in_c * scale),
                    num_filters=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks3"])
            ]
        )

        self.blocks4 = nn.Sequential(
            *[
                DepthwiseSeparable(
                    num_channels=make_divisible(in_c * scale),
                    num_filters=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks4"])
            ]
        )

        self.blocks5 = nn.Sequential(
            *[
                DepthwiseSeparable(
                    num_channels=make_divisible(in_c * scale),
                    num_filters=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks5"])
            ]
        )

        self.blocks6 = nn.Sequential(
            *[
                DepthwiseSeparable(
                    num_channels=make_divisible(in_c * scale),
                    num_filters=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks6"])
            ]
        )

        if pretrained:
            self._load_pretrained(
                MODEL_URLS["PPLCNet_x{}".format(scale)], use_ssld=use_ssld
            )

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        outs.append(x)
        x = self.blocks4(x)
        outs.append(x)
        x = self.blocks5(x)
        outs.append(x)
        x = self.blocks6(x)
        outs.append(x)
        return outs

    def _load_pretrained(self, pretrained_url, use_ssld=False):
        if use_ssld:
            pretrained_url = pretrained_url.replace("_pretrained", "_ssld_pretrained")
        print(pretrained_url)
        local_weight_path = get_path_from_url(
            pretrained_url, os.path.expanduser("~/.paddleclas/weights")
        )
        param_state_dict = paddle.load(local_weight_path)
        self.set_dict(param_state_dict)
        return
