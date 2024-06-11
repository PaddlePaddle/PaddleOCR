# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# The code is based on:
# https://github.com/PaddlePaddle/PaddleDetection/blob/release%2F2.3/ppdet/modeling/necks/csp_pan.py

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr

__all__ = ["CSPPAN"]


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        in_channel=96,
        out_channel=96,
        kernel_size=3,
        stride=1,
        groups=1,
        act="leaky_relu",
    ):
        super(ConvBNLayer, self).__init__()
        initializer = nn.initializer.KaimingUniform()
        self.act = act
        assert self.act in ["leaky_relu", "hard_swish"]
        self.conv = nn.Conv2D(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            groups=groups,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            weight_attr=ParamAttr(initializer=initializer),
            bias_attr=False,
        )
        self.bn = nn.BatchNorm2D(out_channel)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        return x


class DPModule(nn.Layer):
    """
    Depth-wise and point-wise module.
     Args:
        in_channel (int): The input channels of this Module.
        out_channel (int): The output channels of this Module.
        kernel_size (int): The conv2d kernel size of this Module.
        stride (int): The conv2d's stride of this Module.
        act (str): The activation function of this Module,
                   Now support `leaky_relu` and `hard_swish`.
    """

    def __init__(
        self, in_channel=96, out_channel=96, kernel_size=3, stride=1, act="leaky_relu"
    ):
        super(DPModule, self).__init__()
        initializer = nn.initializer.KaimingUniform()
        self.act = act
        self.dwconv = nn.Conv2D(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            groups=out_channel,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            weight_attr=ParamAttr(initializer=initializer),
            bias_attr=False,
        )
        self.bn1 = nn.BatchNorm2D(out_channel)
        self.pwconv = nn.Conv2D(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=1,
            groups=1,
            padding=0,
            weight_attr=ParamAttr(initializer=initializer),
            bias_attr=False,
        )
        self.bn2 = nn.BatchNorm2D(out_channel)

    def act_func(self, x):
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        return x

    def forward(self, x):
        x = self.act_func(self.bn1(self.dwconv(x)))
        x = self.act_func(self.bn2(self.pwconv(x)))
        return x


class DarknetBottleneck(nn.Layer):
    """The basic bottleneck block used in Darknet.
    Each Block consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and act.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.
    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (int): The kernel size of the convolution. Default: 0.5
        add_identity (bool): Whether to add identity to the out.
            Default: True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        expansion=0.5,
        add_identity=True,
        use_depthwise=False,
        act="leaky_relu",
    ):
        super(DarknetBottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        conv_func = DPModule if use_depthwise else ConvBNLayer
        self.conv1 = ConvBNLayer(
            in_channel=in_channels, out_channel=hidden_channels, kernel_size=1, act=act
        )
        self.conv2 = conv_func(
            in_channel=hidden_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=1,
            act=act,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer(nn.Layer):
    """Cross Stage Partial Layer.
    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        num_blocks (int): Number of blocks. Default: 1
        add_identity (bool): Whether to add identity in blocks.
            Default: True
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        expand_ratio=0.5,
        num_blocks=1,
        add_identity=True,
        use_depthwise=False,
        act="leaky_relu",
    ):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvBNLayer(in_channels, mid_channels, 1, act=act)
        self.short_conv = ConvBNLayer(in_channels, mid_channels, 1, act=act)
        self.final_conv = ConvBNLayer(2 * mid_channels, out_channels, 1, act=act)

        self.blocks = nn.Sequential(
            *[
                DarknetBottleneck(
                    mid_channels,
                    mid_channels,
                    kernel_size,
                    1.0,
                    add_identity,
                    use_depthwise,
                    act=act,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = paddle.concat((x_main, x_short), axis=1)
        return self.final_conv(x_final)


class Channel_T(nn.Layer):
    def __init__(self, in_channels=[116, 232, 464], out_channels=96, act="leaky_relu"):
        super(Channel_T, self).__init__()
        self.convs = nn.LayerList()
        for i in range(len(in_channels)):
            self.convs.append(ConvBNLayer(in_channels[i], out_channels, 1, act=act))

    def forward(self, x):
        outs = [self.convs[i](x[i]) for i in range(len(x))]
        return outs


class CSPPAN(nn.Layer):
    """Path Aggregation Network with CSP module.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        kernel_size (int): The conv2d kernel size of this Module.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 1
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: True
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        num_csp_blocks=1,
        use_depthwise=True,
        act="hard_swish",
    ):
        super(CSPPAN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = [out_channels] * len(in_channels)
        conv_func = DPModule if use_depthwise else ConvBNLayer

        self.conv_t = Channel_T(in_channels, out_channels, act=act)

        # build top-down blocks
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.top_down_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                CSPLayer(
                    out_channels * 2,
                    out_channels,
                    kernel_size=kernel_size,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    act=act,
                )
            )

        # build bottom-up blocks
        self.downsamples = nn.LayerList()
        self.bottom_up_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv_func(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    act=act,
                )
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    out_channels * 2,
                    out_channels,
                    kernel_size=kernel_size,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    act=act,
                )
            )

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.
        Returns:
            tuple[Tensor]: CSPPAN features.
        """
        assert len(inputs) == len(self.in_channels)
        inputs = self.conv_t(inputs)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            upsample_feat = F.upsample(
                feat_heigh, size=feat_low.shape[2:4], mode="nearest"
            )

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                paddle.concat([downsample_feat, feat_height], 1)
            )
            outs.append(out)

        return tuple(outs)
