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
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant
from paddle.nn import (
    BatchNorm2D,
    Conv2D,
    GELU,
    Hardsigmoid,
    Hardswish,
    ReLU,
)
from paddle.regularizer import L2Decay


# Network config: small (96->96->192->384, 13 blocks)
NET_CONFIG_SMALL = {
    "blocks2":
    # k, in_c, out_c, s, use_se
    [[3, 96, 96, 1, True]],
    "blocks3": [
        [3, 96, 96, 1, False],
        [3, 96, 96, 1, False],
    ],
    "blocks4": [
        [3, 96, 192, (2, 1), False],
        [3, 192, 192, 1, True],
        [3, 192, 192, 1, False],
        [3, 192, 192, 1, True],
        [3, 192, 192, 1, False],
        [3, 192, 192, 1, True],
        [3, 192, 192, 1, False],
    ],
    "blocks5": [
        [3, 192, 384, (2, 1), False],
        [3, 384, 384, 1, True],
        [3, 384, 384, 1, False],
    ],
    "blocks6": [],
}

# Network config: tiny (48->48->96->160, 10 blocks)
NET_CONFIG_TINY = {
    "blocks2": [[3, 48, 48, 1, True]],
    "blocks3": [
        [3, 48, 48, 1, False],
    ],
    "blocks4": [
        [3, 48, 96, (2, 1), False],
        [3, 96, 96, 1, True],
        [3, 96, 96, 1, False],
    ],
    "blocks5": [
        [3, 96, 160, (2, 1), False],
        [3, 160, 160, 1, True],
        [3, 160, 160, 1, False],
        [3, 160, 160, 1, False],
    ],
    "blocks6": [],
}

# Network config: base (128->128->384->640, 15 blocks, 6 SE)
NET_CONFIG_BASE = {
    # k, in_c, out_c, s, use_se
    "blocks2": [[3, 128, 128, 1, True]],
    "blocks3": [
        [3, 128, 128, 1, False],
        [3, 128, 128, 1, False],
    ],
    "blocks4": [
        [3, 128, 384, (2, 1), False],
        [3, 384, 384, 1, True],
        [3, 384, 384, 1, False],
        [3, 384, 384, 1, True],
        [3, 384, 384, 1, False],
        [3, 384, 384, 1, True],
        [3, 384, 384, 1, False],
        [3, 384, 384, 1, False],
    ],
    "blocks5": [
        [3, 384, 640, (2, 1), False],
        [3, 640, 640, 1, True],
        [3, 640, 640, 1, False],
        [3, 640, 640, 1, True],
    ],
    "blocks6": [],
}


class Conv2D_BN(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        bn_weight_init=1.0,
    ):
        super().__init__()
        self.add_sublayer(
            "conv",
            Conv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias_attr=False,
            ),
        )
        bn = BatchNorm2D(out_channels)
        if bn_weight_init == 1.0:
            Constant(1.0)(bn.weight)
        else:
            Constant(0.0)(bn.weight)
        Constant(0.0)(bn.bias)
        self.add_sublayer("bn", bn)

    @paddle.no_grad()
    def fuse(self):
        c, bn = self.conv, self.bn
        w = bn.weight / (bn._variance + bn._epsilon) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn._mean * bn.weight / (bn._variance + bn._epsilon) ** 0.5
        m = Conv2D(
            w.shape[1] * c._groups,
            w.shape[0],
            w.shape[2:],
            stride=c._stride,
            padding=c._padding,
            groups=c._groups,
        )
        m.weight.set_value(w)
        m.bias.set_value(b)
        return m


class ConvBNAct(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        use_act=True,
        lr_mult=1.0,
    ):
        super().__init__()
        self.use_act = use_act
        self.conv = Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding if isinstance(padding, str) else (kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=False,
        )
        self.bn = BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0), learning_rate=lr_mult),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0), learning_rate=lr_mult),
        )
        if self.use_act:
            self.act = ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x


class StemBlock(nn.Layer):
    """Multi-branch stem with total stride 4 (stem1 stride=2 + stem3 stride=2)."""

    def __init__(
        self,
        in_channels=3,
        mid_channels=48,
        out_channels=96,
        lr_mult=1.0,
    ):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_act=True,
            lr_mult=lr_mult,
        )
        self.stem2a = ConvBNAct(
            in_channels=mid_channels,
            out_channels=mid_channels // 2,
            kernel_size=2,
            stride=1,
            padding="SAME",
            use_act=True,
            lr_mult=lr_mult,
        )
        self.stem2b = ConvBNAct(
            in_channels=mid_channels // 2,
            out_channels=mid_channels,
            kernel_size=2,
            stride=1,
            padding="SAME",
            use_act=True,
            lr_mult=lr_mult,
        )
        self.stem3 = ConvBNAct(
            in_channels=mid_channels * 2,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_act=True,
            lr_mult=lr_mult,
        )
        self.stem4 = ConvBNAct(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_act=True,
            lr_mult=lr_mult,
        )
        self.pool = nn.MaxPool2D(
            kernel_size=2, stride=1, ceil_mode=True, padding="SAME"
        )

    def forward(self, x):
        x = self.stem1(x)
        x2 = self.stem2a(x)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = paddle.concat([x1, x2], axis=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class SELayer(nn.Layer):
    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult),
        )
        self.relu = ReLU()
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult),
        )
        self.hardsigmoid = Hardsigmoid()

    def forward(self, x):
        identity = x
        x = x.mean(axis=[2, 3], keepdim=True)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class RepDWConv(nn.Layer):
    """Reparameterizable depthwise convolution.

    Training: 3-branch structure (3x3 DW + 1x1 DW + identity)
    Inference: fused into a single 3x3 DW Conv
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2

        self.conv = Conv2D_BN(
            channels, channels, kernel_size, 1, padding, groups=channels
        )
        self.conv1 = Conv2D(
            channels, channels, 1, 1, 0, groups=channels, bias_attr=False
        )
        self.bn = BatchNorm2D(channels)
        Constant(1.0)(self.bn.weight)
        Constant(0.0)(self.bn.bias)

        self.is_repped = False
        self.reparam_conv = None

    def forward(self, x):
        if self.is_repped:
            return self.reparam_conv(x)
        return self.bn(self.conv(x) + self.conv1(x) + x)

    def rep(self):
        if self.is_repped:
            return

        fused_conv = self._fuse_conv()

        padding = (self.kernel_size - 1) // 2
        self.reparam_conv = Conv2D(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            groups=self.channels,
        )
        self.reparam_conv.weight.set_value(fused_conv.weight)
        self.reparam_conv.bias.set_value(fused_conv.bias)

        self.__delattr__("conv")
        self.__delattr__("conv1")
        self.__delattr__("bn")

        self.is_repped = True

    @paddle.no_grad()
    def _fuse_conv(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight

        pad_size = self.kernel_size // 2
        conv1_w = F.pad(conv1_w, [pad_size, pad_size, pad_size, pad_size])

        identity = F.pad(
            paddle.ones([conv1_w.shape[0], conv1_w.shape[1], 1, 1]),
            [pad_size, pad_size, pad_size, pad_size],
        )

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b

        conv.weight.set_value(final_conv_w)
        conv.bias.set_value(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn._variance + bn._epsilon) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = (
            bn.bias
            + (conv.bias - bn._mean) * bn.weight / (bn._variance + bn._epsilon) ** 0.5
        )

        conv.weight.set_value(w)
        conv.bias.set_value(b)
        return conv

    def fuse(self):
        return self._fuse_conv()


class LCNetV4Block(nn.Layer):
    """LCNetV4 Block: DW Conv -> [SE] -> ChannelMixer(expand->act->compress) + residual"""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dw_size,
        use_se=False,
        lr_mult=1.0,
        expand_ratio=2,
        act_type="gelu",
    ):
        super().__init__()
        self.has_residual = in_channels == out_channels and stride == 1

        padding = (dw_size - 1) // 2
        self.token_mixer = nn.Sequential()
        self.token_mixer.add_sublayer(
            "dw_conv",
            Conv2D_BN(
                in_channels, in_channels, dw_size, stride, padding, groups=in_channels
            ),
        )
        if use_se:
            self.token_mixer.add_sublayer("se", SELayer(in_channels, lr_mult=lr_mult))

        hidden_channels = int(in_channels * expand_ratio)
        compress_bn_init = 0.0 if self.has_residual else 1.0
        self.channel_mixer = nn.Sequential()
        self.channel_mixer.add_sublayer(
            "expand", Conv2D_BN(in_channels, hidden_channels, 1, 1, 0)
        )
        if act_type == "gelu":
            self.channel_mixer.add_sublayer("act", GELU())
        elif act_type == "hswish":
            self.channel_mixer.add_sublayer("act", Hardswish())
        elif act_type == "relu":
            self.channel_mixer.add_sublayer("act", ReLU())
        self.channel_mixer.add_sublayer(
            "compress",
            Conv2D_BN(
                hidden_channels, out_channels, 1, 1, 0, bn_weight_init=compress_bn_init
            ),
        )

    def forward(self, x):
        x = self.token_mixer(x)
        if self.has_residual:
            return x + self.channel_mixer(x)
        else:
            return self.channel_mixer(x)


class LCNetV4RepBlock(nn.Layer):
    """LCNetV4 Rep Block: RepDWConv -> [SE] -> ChannelMixer + residual

    Uses RepDWConv (3-branch training, fused inference) when stride=1 and channels unchanged.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dw_size,
        use_se=False,
        lr_mult=1.0,
        expand_ratio=2,
        act_type="gelu",
    ):
        super().__init__()
        self.has_residual = in_channels == out_channels and stride == 1
        self.use_rep_dw = stride == 1 and in_channels == out_channels

        self.token_mixer = nn.Sequential()
        if self.use_rep_dw:
            self.token_mixer.add_sublayer("rep_dw", RepDWConv(in_channels, dw_size))
        else:
            padding = (dw_size - 1) // 2
            self.token_mixer.add_sublayer(
                "dw_conv",
                Conv2D_BN(
                    in_channels,
                    in_channels,
                    dw_size,
                    stride,
                    padding,
                    groups=in_channels,
                ),
            )
        if use_se:
            self.token_mixer.add_sublayer("se", SELayer(in_channels, lr_mult=lr_mult))

        hidden_channels = int(in_channels * expand_ratio)
        compress_bn_init = 0.0 if self.has_residual else 1.0
        self.channel_mixer = nn.Sequential()
        self.channel_mixer.add_sublayer(
            "expand", Conv2D_BN(in_channels, hidden_channels, 1, 1, 0)
        )
        if act_type == "gelu":
            self.channel_mixer.add_sublayer("act", GELU())
        elif act_type == "hswish":
            self.channel_mixer.add_sublayer("act", Hardswish())
        elif act_type == "relu":
            self.channel_mixer.add_sublayer("act", ReLU())
        self.channel_mixer.add_sublayer(
            "compress",
            Conv2D_BN(
                hidden_channels, out_channels, 1, 1, 0, bn_weight_init=compress_bn_init
            ),
        )

    def forward(self, x):
        x = self.token_mixer(x)
        if self.has_residual:
            return x + self.channel_mixer(x)
        else:
            return self.channel_mixer(x)

    def rep(self):
        if hasattr(self, "is_repped") and self.is_repped:
            return
        if self.use_rep_dw and hasattr(self.token_mixer, "rep_dw"):
            self.token_mixer.rep_dw.rep()
        self.is_repped = True


class PPLCNetV4(nn.Layer):
    """PPLCNetV4 Backbone for PP-OCRv6 text recognition.

    Three model variants:
    - base:  128->128->384->640 channels, branch stem, output [B, 640, 1, 40]
    - small: 96->96->192->384 channels, branch stem, output [B, 384, 1, 40]
    - tiny:  48->48->96->160 channels, simple stem, output [B, 160, 1, 40]

    Args:
        lr_mult_list (list): Learning rate multipliers for each stage.
        use_rep (bool): Use RepDWConv in TokenMixer (3-branch training, fused inference).
        config_name (str): 'small' (default), 'tiny', or 'base'.
        stem_type (str): 'simple' (two-layer Conv) or 'branch' (multi-branch StemBlock).
        stem_channels (int): Stem output channels (128 for base, 96 for small, 48 for tiny).
    """

    def __init__(
        self,
        lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        use_rep=False,
        config_name="small",
        stem_type="simple",
        stem_channels=96,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(lr_mult_list, (list, tuple))
        assert len(lr_mult_list) == 6
        self.lr_mult_list = lr_mult_list
        self.use_rep = use_rep

        if config_name == "tiny":
            self.net_config = NET_CONFIG_TINY
        elif config_name == "base":
            self.net_config = NET_CONFIG_BASE
        else:
            self.net_config = NET_CONFIG_SMALL

        if stem_type == "branch":
            self.conv1 = StemBlock(
                in_channels=3,
                mid_channels=stem_channels // 2,
                out_channels=stem_channels,
                lr_mult=lr_mult_list[0],
            )
        else:
            self.conv1 = nn.Sequential(
                Conv2D_BN(3, stem_channels // 2, 3, 2, 1),
                GELU(),
                Conv2D_BN(stem_channels // 2, stem_channels, 3, 2, 1),
            )

        self.blocks2 = self._make_stage("blocks2", 1)
        self.blocks3 = self._make_stage("blocks3", 2)
        self.blocks4 = self._make_stage("blocks4", 3)
        self.blocks5 = self._make_stage("blocks5", 4)
        self.blocks6 = self._make_stage("blocks6", 5)

        for stage_name in reversed(
            ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
        ):
            stage_cfg = self.net_config.get(stage_name, [])
            if stage_cfg:
                self.out_channels = stage_cfg[-1][2]
                break
        else:
            self.out_channels = stem_channels

    def _make_stage(self, stage_name, lr_mult_idx):
        blocks = []
        stage_config = self.net_config.get(stage_name, [])

        for config in stage_config:
            k, in_c, out_c, s, se = config
            if self.use_rep:
                blocks.append(
                    LCNetV4RepBlock(
                        in_channels=in_c,
                        out_channels=out_c,
                        dw_size=k,
                        stride=s,
                        use_se=se,
                        lr_mult=self.lr_mult_list[lr_mult_idx],
                        expand_ratio=2,
                    )
                )
            else:
                blocks.append(
                    LCNetV4Block(
                        in_channels=in_c,
                        out_channels=out_c,
                        dw_size=k,
                        stride=s,
                        use_se=se,
                        lr_mult=self.lr_mult_list[lr_mult_idx],
                        expand_ratio=2,
                    )
                )
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)

        if self.training:
            x = F.adaptive_avg_pool2d(x, [1, 40])
        else:
            assert x.shape[2] >= 3, (
                f"Feature height {x.shape[2]} < pool kernel 3. "
                f"Check spatial downsampling config with current stem."
            )
            x = F.avg_pool2d(x, [3, 2])
        return x

    def rep(self):
        if not self.use_rep:
            return
        for blocks in [
            self.blocks2,
            self.blocks3,
            self.blocks4,
            self.blocks5,
            self.blocks6,
        ]:
            for block in blocks:
                if hasattr(block, "rep"):
                    block.rep()
