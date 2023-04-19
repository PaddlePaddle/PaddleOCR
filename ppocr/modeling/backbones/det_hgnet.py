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
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm2D, ReLU, AdaptiveAvgPool2D, MaxPool2D
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Normal, Constant, XavierUniform, KaimingNormal

__all__ = ["PPHGNet"]

SCALE_CONFIG = {
    "small": {
        "stage_config": {
            # in_channels, mid_channels, out_channels, blocks, downsample
            "stage1": [128, 128, 256, 1, False],
            "stage2": [256, 160, 512, 1, True],
            "stage3": [512, 192, 768, 2, True],
            "stage4": [768, 224, 1024, 1, True],
        },
        "stem_channels": [64, 64, 128],
        "layer_num": 6,
    },
    "tiny": {
        "stage_config": {
            # in_channels, mid_channels, out_channels, blocks, downsample
            "stage1": [96, 96, 224, 1, False],
            "stage2": [224, 128, 448, 1, True],
            "stage3": [448, 160, 512, 2, True],
            "stage4": [512, 192, 768, 1, True],
        },
        "stem_channels": [48, 48, 96],
        "layer_num": 5,
    }
}

kaiming_normal_ = KaimingNormal()
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


class ConvBNAct(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 use_act=True):
        super().__init__()
        self.use_act = use_act
        self.conv = Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias_attr=False)
        self.bn = BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        if self.use_act:
            self.act = ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x


class ESEModule(nn.Layer):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv = Conv2D(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return paddle.multiply(x=identity, y=x)


class HG_Block(nn.Layer):
    def __init__(
            self,
            in_channels,
            mid_channels,
            out_channels,
            layer_num,
            identity=False, ):
        super().__init__()
        self.identity = identity

        self.layers = nn.LayerList()
        self.layers.append(
            ConvBNAct(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1))
        for _ in range(layer_num - 1):
            self.layers.append(
                ConvBNAct(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    stride=1))

        # feature aggregation
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_conv = ConvBNAct(
            in_channels=total_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1)
        self.att = ESEModule(out_channels)

    def forward(self, x):
        identity = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = paddle.concat(output, axis=1)
        x = self.aggregation_conv(x)
        x = self.att(x)
        if self.identity:
            x += identity
        return x


class HG_Stage(nn.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 block_num,
                 layer_num,
                 downsample=True):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
                use_act=False)

        blocks_list = []
        blocks_list.append(
            HG_Block(
                in_channels,
                mid_channels,
                out_channels,
                layer_num,
                identity=False))
        for _ in range(block_num - 1):
            blocks_list.append(
                HG_Block(
                    out_channels,
                    mid_channels,
                    out_channels,
                    layer_num,
                    identity=True))
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


class PPHGNet(nn.Layer):
    """
    PPHGNet
    Args:
        stem_channels: list. Stem channel list of PPHGNet.
        stage_config: dict. The configuration of each stage of PPHGNet. such as the number of channels, stride, etc.
        layer_num: int. Number of layers of HG_Block.
        use_last_conv: boolean. Whether to use a 1x1 convolutional layer before the classification layer.
        class_expand: int=2048. Number of channels for the last 1x1 convolutional layer.
        dropout_prob: float. Parameters of dropout, 0.0 means dropout is not used.
        class_num: int=1000. The number of classes.
    Returns:
        model: nn.Layer. Specific PPHGNet model depends on args.
    """

    def __init__(
            self,
            in_channels,
            scale,
            use_last_conv=True,
            class_expand=2048,
            dropout_prob=0.0,
            class_num=1000,
            out_indices=None, ):
        super().__init__()
        self.use_last_conv = use_last_conv
        self.class_expand = class_expand
        stage_config = SCALE_CONFIG[scale]["stage_config"]
        stem_channels = SCALE_CONFIG[scale]["stem_channels"]
        layer_num = SCALE_CONFIG[scale]["layer_num"]
        self.out_indices = out_indices if out_indices is not None else [
            0, 1, 2, 3
        ]
        # stem
        stem_channels.insert(0, in_channels)
        self.stem = nn.Sequential(* [
            ConvBNAct(
                in_channels=stem_channels[i],
                out_channels=stem_channels[i + 1],
                kernel_size=3,
                stride=2 if i == 0 else 1) for i in range(
                    len(stem_channels) - 1)
        ])
        self.pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # stages
        self.stages = nn.LayerList()
        self.out_channels = []
        for block, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample = stage_config[
                k]
            self.stages.append(
                HG_Stage(in_channels, mid_channels, out_channels, block_num,
                         layer_num, downsample))
            if block in self.out_indices:
                self.out_channels.append(out_channels)

        self.avg_pool = AdaptiveAvgPool2D(1)
        if self.use_last_conv:
            self.last_conv = Conv2D(
                in_channels=out_channels,
                out_channels=self.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False)
            self.act = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout_prob, mode="downscale_in_infer")

        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.fc = nn.Linear(self.class_expand
                            if self.use_last_conv else out_channels, class_num)

        self._init_weights()

    def _init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2D)):
                ones_(m.weight)
                zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)

        out = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                out.append(x)
        return out
