# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingNormal, Constant
from paddle.nn import Conv2D, BatchNorm2D, ReLU, AdaptiveAvgPool2D, MaxPool2D
from paddle.regularizer import L2Decay
from paddle import ParamAttr

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
                 downsample=True,
                 stride=[2, 1]):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=stride,
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
            stem_channels,
            stage_config,
            layer_num,
            in_channels=3,
            det=False,
            out_indices=None, ):
        super().__init__()
        self.det = det
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

        if self.det:
            self.pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        # stages
        self.stages = nn.LayerList()
        self.out_channels = []
        for block_id, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, stride = stage_config[
                k]
            self.stages.append(
                HG_Stage(in_channels, mid_channels, out_channels, block_num,
                         layer_num, downsample, stride))
            if block_id in self.out_indices:
                self.out_channels.append(out_channels)

        if not self.det:
            self.out_channels = stage_config["stage4"][2]

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
        if self.det:
            x = self.pool(x)

        out = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.det and i in self.out_indices:
                out.append(x)
        if self.det:
            return out

        if self.training:
            x = F.adaptive_avg_pool2d(x, [1, 40])
        else:
            x = F.avg_pool2d(x, [3, 2])
        return x


def PPHGNet_tiny(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNet_tiny
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPHGNet_tiny` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, blocks, downsample
        "stage1": [96, 96, 224, 1, False, [2, 1]],
        "stage2": [224, 128, 448, 1, True, [1, 2]],
        "stage3": [448, 160, 512, 2, True, [2, 1]],
        "stage4": [512, 192, 768, 1, True, [2, 1]],
    }

    model = PPHGNet(
        stem_channels=[48, 48, 96],
        stage_config=stage_config,
        layer_num=5,
        **kwargs)
    return model


def PPHGNet_small(pretrained=False, use_ssld=False, det=False, **kwargs):
    """
    PPHGNet_small
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPHGNet_small` model depends on args.
    """
    stage_config_det = {
        # in_channels, mid_channels, out_channels, blocks, downsample
        "stage1": [128, 128, 256, 1, False, 2],
        "stage2": [256, 160, 512, 1, True, 2],
        "stage3": [512, 192, 768, 2, True, 2],
        "stage4": [768, 224, 1024, 1, True, 2],
    }

    stage_config_rec = {
        # in_channels, mid_channels, out_channels, blocks, downsample
        "stage1": [128, 128, 256, 1, True, [2, 1]],
        "stage2": [256, 160, 512, 1, True, [1, 2]],
        "stage3": [512, 192, 768, 2, True, [2, 1]],
        "stage4": [768, 224, 1024, 1, True, [2, 1]],
    }

    model = PPHGNet(
        stem_channels=[64, 64, 128],
        stage_config=stage_config_det if det else stage_config_rec,
        layer_num=6,
        det=det,
        **kwargs)
    return model


def PPHGNet_base(pretrained=False, use_ssld=True, **kwargs):
    """
    PPHGNet_base
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPHGNet_base` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, blocks, downsample
        "stage1": [160, 192, 320, 1, False, [2, 1]],
        "stage2": [320, 224, 640, 2, True, [1, 2]],
        "stage3": [640, 256, 960, 3, True, [2, 1]],
        "stage4": [960, 288, 1280, 2, True, [2, 1]],
    }

    model = PPHGNet(
        stem_channels=[96, 96, 160],
        stage_config=stage_config,
        layer_num=7,
        dropout_prob=0.2,
        **kwargs)
    return model
