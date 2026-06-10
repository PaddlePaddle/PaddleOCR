# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
import os
import sys
from ppocr.modeling.necks.intracl import IntraCLBlock

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../..")))

from ppocr.modeling.backbones.det_mobilenet_v3 import SEModule


class DSConv(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride=1,
        groups=None,
        if_act=True,
        act="relu",
        **kwargs,
    ):
        super(DSConv, self).__init__()
        if groups == None:
            groups = in_channels
        self.if_act = if_act
        self.act = act
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False,
        )

        self.bn1 = nn.BatchNorm(num_channels=in_channels, act=None)

        self.conv2 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=int(in_channels * 4),
            kernel_size=1,
            stride=1,
            bias_attr=False,
        )

        self.bn2 = nn.BatchNorm(num_channels=int(in_channels * 4), act=None)

        self.conv3 = nn.Conv2D(
            in_channels=int(in_channels * 4),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias_attr=False,
        )
        self._c = [in_channels, out_channels]
        if in_channels != out_channels:
            self.conv_end = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias_attr=False,
            )

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print(
                    "The activation function({}) is selected incorrectly.".format(
                        self.act
                    )
                )
                exit()

        x = self.conv3(x)
        if self._c[0] != self._c[1]:
            x = x + self.conv_end(inputs)
        return x


class DBFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, use_asf=False, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        self.use_asf = use_asf
        weight_attr = paddle.nn.initializer.KaimingUniform()

        self.in2_conv = nn.Conv2D(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.in3_conv = nn.Conv2D(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.in4_conv = nn.Conv2D(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.in5_conv = nn.Conv2D(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.p5_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.p4_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.p3_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.p2_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )

        if self.use_asf is True:
            self.asf = ASFBlock(self.out_channels, self.out_channels // 4)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/4

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)
        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)

        if self.use_asf is True:
            fuse = self.asf(fuse, [p5, p4, p3, p2])

        return fuse


class RSELayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super(RSELayer, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.se_block = SEModule(self.out_channels)
        self.shortcut = shortcut

    def forward(self, ins):
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out


class RSEFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, shortcut=True, **kwargs):
        super(RSEFPN, self).__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.LayerList()
        self.inp_conv = nn.LayerList()
        self.intracl = False
        if "intracl" in kwargs.keys() and kwargs["intracl"] is True:
            self.intracl = kwargs["intracl"]
            self.incl1 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)

        for i in range(len(in_channels)):
            self.ins_conv.append(
                RSELayer(in_channels[i], out_channels, kernel_size=1, shortcut=shortcut)
            )
            self.inp_conv.append(
                RSELayer(
                    out_channels, out_channels // 4, kernel_size=3, shortcut=shortcut
                )
            )

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/4

        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)

        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        return fuse


class RepLKFPN(nn.Layer):
    """Optimized RSEFPN: replaces 3x3 standard Conv in inp_conv with
    DilatedReparamBlock (DW, 5x5) + PWConv 1x1 + SE.

    Uses the existing DilatedReparamBlock from UniRepLKNet to provide
    multi-branch dilated training with single-conv inference.

    Changes vs RSEFPN:
      - inp_conv: RSELayer(3x3 std Conv + SE)
                → DilatedReparamBlock(5x5 DW) + PWConv(1x1) + SE
      - ins_conv: unchanged (1x1 Conv, no benefit from DW decomposition)

    Parameter comparison (out_channels=96, 4 levels):
      RSEFPN  inp_conv: 4 × (96×24×9 + SE) = 4 × 21,054 = 84,216
      RepLKFPN inp_conv: 4 × (DilReparam96 + 96×24 + SE)
             = 4 × (96×25 + 96×2 + 2×(96×9+96×2) + 96×24 + SE)
             = 4 × (2,400 + 192 + 2×(864+192) + 2,304 + 318)
             = 4 × 7,326 = 29,304
      inp_conv reduction: 65.2%
      Receptive field: 3×3 → 5×5 (with multi-dilation 3,5 patterns in training)

    Inference: DilatedReparamBlock merges to single 5x5 DWConv, zero extra cost.
    """

    def __init__(
        self, in_channels, out_channels, shortcut=True, dilated_kernel_size=7, **kwargs
    ):
        super(RepLKFPN, self).__init__()
        self.out_channels = out_channels
        self.is_repped = False
        self.ins_conv = nn.LayerList()
        self.inp_conv_dw = nn.LayerList()
        self.inp_conv_pw = nn.LayerList()
        self.inp_conv_se = nn.LayerList()
        self.shortcut = shortcut

        self.intracl = False
        if "intracl" in kwargs.keys() and kwargs["intracl"] is True:
            self.intracl = kwargs["intracl"]
            self.incl1 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)

        weight_attr = paddle.nn.initializer.KaimingUniform()

        for i in range(len(in_channels)):
            self.ins_conv.append(
                RSELayer(in_channels[i], out_channels, kernel_size=1, shortcut=shortcut)
            )

            self.inp_conv_dw.append(
                DilatedReparamBlock(
                    channels=out_channels, kernel_size=dilated_kernel_size
                )
            )

            self.inp_conv_pw.append(
                nn.Conv2D(
                    in_channels=out_channels,
                    out_channels=out_channels // 4,
                    kernel_size=1,
                    weight_attr=ParamAttr(initializer=weight_attr),
                    bias_attr=False,
                )
            )

            self.inp_conv_se.append(SEModule(out_channels // 4))

    def _inp_forward(self, x, idx):
        x = self.inp_conv_dw[idx](x)
        x = self.inp_conv_pw[idx](x)
        if self.shortcut:
            x = x + self.inp_conv_se[idx](x)
        else:
            x = self.inp_conv_se[idx](x)
        return x

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/4

        p5 = self._inp_forward(in5, 3)
        p4 = self._inp_forward(out4, 2)
        p3 = self._inp_forward(out3, 1)
        p2 = self._inp_forward(out2, 0)

        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        if self.training:
            return {"fuse": fuse, "aux_p4": out4, "aux_p3": out3, "aux_p2": out2}
        return fuse

    def rep(self):
        """Merge DilatedReparamBlock branches for inference deployment."""
        if self.is_repped:
            return
        for i in range(len(self.inp_conv_dw)):
            self.inp_conv_dw[i].rep()
        self.is_repped = True


class LKPAN(nn.Layer):
    def __init__(self, in_channels, out_channels, mode="large", **kwargs):
        super(LKPAN, self).__init__()
        self.out_channels = out_channels
        weight_attr = paddle.nn.initializer.KaimingUniform()

        self.ins_conv = nn.LayerList()
        self.inp_conv = nn.LayerList()
        # pan head
        self.pan_head_conv = nn.LayerList()
        self.pan_lat_conv = nn.LayerList()

        if mode.lower() == "lite":
            p_layer = DSConv
        elif mode.lower() == "large":
            p_layer = nn.Conv2D
        else:
            raise ValueError(
                "mode can only be one of ['lite', 'large'], but received {}".format(
                    mode
                )
            )

        for i in range(len(in_channels)):
            self.ins_conv.append(
                nn.Conv2D(
                    in_channels=in_channels[i],
                    out_channels=self.out_channels,
                    kernel_size=1,
                    weight_attr=ParamAttr(initializer=weight_attr),
                    bias_attr=False,
                )
            )

            self.inp_conv.append(
                p_layer(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                    padding=4,
                    weight_attr=ParamAttr(initializer=weight_attr),
                    bias_attr=False,
                )
            )

            if i > 0:
                self.pan_head_conv.append(
                    nn.Conv2D(
                        in_channels=self.out_channels // 4,
                        out_channels=self.out_channels // 4,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        weight_attr=ParamAttr(initializer=weight_attr),
                        bias_attr=False,
                    )
                )
            self.pan_lat_conv.append(
                p_layer(
                    in_channels=self.out_channels // 4,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                    padding=4,
                    weight_attr=ParamAttr(initializer=weight_attr),
                    bias_attr=False,
                )
            )

        self.intracl = False
        if "intracl" in kwargs.keys() and kwargs["intracl"] is True:
            self.intracl = kwargs["intracl"]
            self.incl1 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/4

        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)

        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)

        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)

        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        return fuse


class DilatedReparamBlock(nn.Layer):
    """
    Dilated Reparam Block from UniRepLKNet.
    Reference: https://github.com/AILab-CVC/UniRepLKNet

    Training: uses multiple parallel dilated depthwise convolutions + BN
    Inference: all branches merge into a single large-kernel depthwise conv

    For kernel_size=9, the branches are:
      - origin: 9x9 DW Conv (dil=1)
      - branch1: 5x5 DW Conv (dil=1, equiv RF=5)
      - branch2: 5x5 DW Conv (dil=2, equiv RF=9)
      - branch3: 3x3 DW Conv (dil=3, equiv RF=7)
      - branch4: 3x3 DW Conv (dil=4, equiv RF=9)
    """

    def __init__(self, channels, kernel_size=9, deploy=False):
        super(DilatedReparamBlock, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.is_repped = deploy

        if kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        else:
            raise ValueError(
                "DilatedReparamBlock requires kernel_size in [5,7,9,11,13], "
                "but got {}".format(kernel_size)
            )

        if not self.is_repped:
            self.lk_origin = nn.Conv2D(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=channels,
                bias_attr=False,
            )
            self.origin_bn = nn.BatchNorm2D(channels)

            for k, r in zip(self.kernel_sizes, self.dilates):
                equiv_ks = r * (k - 1) + 1
                p = equiv_ks // 2
                conv = nn.Conv2D(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=k,
                    stride=1,
                    padding=p,
                    dilation=r,
                    groups=channels,
                    bias_attr=False,
                )
                bn = nn.BatchNorm2D(channels)
                setattr(self, "dil_conv_k{}_{}".format(k, r), conv)
                setattr(self, "dil_bn_k{}_{}".format(k, r), bn)
        else:
            self.lk_origin = nn.Conv2D(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=channels,
                bias_attr=True,
            )

    def forward(self, x):
        if self.is_repped:
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, "dil_conv_k{}_{}".format(k, r))
            bn = getattr(self, "dil_bn_k{}_{}".format(k, r))
            out = out + bn(conv(x))
        return out

    @staticmethod
    def _fuse_bn(conv, bn):
        """Fuse Conv2D + BatchNorm2D into a single Conv2D (weight, bias)."""
        kernel = conv.weight
        gamma = bn.weight
        beta = bn.bias
        running_mean = bn._mean
        running_var = bn._variance
        eps = bn._epsilon
        std = paddle.sqrt(running_var + eps)
        fused_weight = kernel * (gamma / std).reshape([-1, 1, 1, 1])
        fused_bias = beta - running_mean * gamma / std
        return fused_weight, fused_bias

    @staticmethod
    def _convert_dilated_to_nondilated(kernel, dilate_rate):
        """Convert dilated conv kernel to equivalent non-dilated (sparse) kernel
        by inserting zeros between kernel elements using transposed convolution."""
        if dilate_rate == 1:
            return kernel
        identity = paddle.ones(shape=[1, 1, 1, 1], dtype=kernel.dtype)
        # F.conv2d_transpose with stride=dilate_rate inserts zeros
        # Process each channel independently
        C = kernel.shape[0]
        result_list = []
        for i in range(C):
            k_i = kernel[i : i + 1]  # (1, 1, kH, kW)
            dilated = F.conv2d_transpose(k_i, identity, stride=dilate_rate)
            result_list.append(dilated)
        return paddle.concat(result_list, axis=0)

    @staticmethod
    def _merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
        """Pad dilated equivalent kernel to large kernel size and add."""
        large_k = large_kernel.shape[2]
        dilated_k = dilated_kernel.shape[2]
        equiv_ks = dilated_r * (dilated_k - 1) + 1
        equiv_kernel = DilatedReparamBlock._convert_dilated_to_nondilated(
            dilated_kernel, dilated_r
        )
        rows_to_pad = large_k // 2 - equiv_ks // 2
        if rows_to_pad > 0:
            merged = large_kernel + F.pad(
                equiv_kernel, [rows_to_pad, rows_to_pad, rows_to_pad, rows_to_pad]
            )
        else:
            merged = large_kernel + equiv_kernel
        return merged

    @paddle.no_grad()
    def rep(self):
        """Merge all parallel branches into a single large-kernel DW conv.
        Call this before switching to deploy/inference mode."""
        if self.is_repped:
            return
        origin_k, origin_b = self._fuse_bn(self.lk_origin, self.origin_bn)
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, "dil_conv_k{}_{}".format(k, r))
            bn = getattr(self, "dil_bn_k{}_{}".format(k, r))
            branch_k, branch_b = self._fuse_bn(conv, bn)
            origin_k = self._merge_dilated_into_large_kernel(origin_k, branch_k, r)
            origin_b = origin_b + branch_b

        merged_conv = nn.Conv2D(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.channels,
            bias_attr=True,
        )
        merged_conv.weight.set_value(origin_k)
        merged_conv.bias.set_value(origin_b)
        self.lk_origin = merged_conv
        self.is_repped = True

        delattr(self, "origin_bn")
        for k, r in zip(self.kernel_sizes, self.dilates):
            delattr(self, "dil_conv_k{}_{}".format(k, r))
            delattr(self, "dil_bn_k{}_{}".format(k, r))


class DilatedReparamConv(nn.Layer):
    """
    A drop-in replacement for standard Conv2D (in_ch → out_ch, large kernel)
    using DilatedReparamBlock (depthwise) + 1x1 pointwise convolution.

    Architecture:
      input(in_ch) → DilatedReparamBlock(in_ch, DW, kernel_size) → 1x1 Conv(in_ch→out_ch) → BN

    This decomposition replaces a single large standard conv with DW + PW,
    drastically reducing parameters while maintaining the large receptive field.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=9, deploy=False, **kwargs
    ):
        super(DilatedReparamConv, self).__init__()
        self.is_repped = False
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.dw = DilatedReparamBlock(
            channels=in_channels, kernel_size=kernel_size, deploy=deploy
        )
        self.pw = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        if not self.is_repped:
            x = self.bn(x)
        return x

    @paddle.no_grad()
    def rep(self):
        """Fuse DW branches + PW Conv + BN for deployment."""
        if self.is_repped:
            return
        self.dw.rep()
        # Fuse pw(Conv2D, no bias) + bn(BatchNorm2D) into single Conv2D with bias
        conv, bn = self.pw, self.bn
        gamma = bn.weight
        std = paddle.sqrt(bn._variance + bn._epsilon)
        scale = gamma / std
        w = conv.weight * scale[:, None, None, None]
        b = bn.bias - bn._mean * scale
        fused = nn.Conv2D(
            conv._in_channels,
            conv._out_channels,
            conv._kernel_size,
            stride=conv._stride,
            padding=conv._padding,
            dilation=conv._dilation,
            groups=conv._groups,
        )
        fused.weight.set_value(w)
        fused.bias.set_value(b)
        self.pw = fused
        del self.bn
        self.is_repped = True


class RepLKPAN(nn.Layer):
    """
    Optimized LKPAN using UniRepLKNet's DilatedReparamBlock.

    Replaces the 8 standard 9x9 Conv2D in LKPAN (4 inp_conv + 4 pan_lat_conv)
    with DilatedReparamConv (DW large-kernel reparam + 1x1 pointwise).

    Parameter comparison (out_channels=256, i.e. inner_ch=64):
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ Component        │ Original (Std 9x9 Conv)     │ UniRepLK (DW+PW)       │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ inp_conv[i]:     │ 256×64×9×9 = 1,327,104      │ DW: 256×1×9×9 = 20,736 │
    │ (256→64, ×4)     │                              │  +4 dilated DW branches │
    │                  │                              │  ≈ 256×(25+25+9+9)     │
    │                  │                              │  = 17,408 (training)    │
    │                  │                              │ PW: 256×64×1×1 = 16,384 │
    │                  │                              │ BN: 64×2 = 128          │
    │                  │                              │ Subtotal/layer ≈ 54,656 │
    │                  │                              │ (vs 1,327,104 original) │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ pan_lat_conv[i]: │ 64×64×9×9 = 331,776         │ DW: 64×1×9×9 = 5,184   │
    │ (64→64, ×4)      │                              │  +4 dilated DW branches │
    │                  │                              │  ≈ 64×(25+25+9+9)      │
    │                  │                              │  = 4,352 (training)     │
    │                  │                              │ PW: 64×64×1×1 = 4,096   │
    │                  │                              │ BN: 64×2 = 128          │
    │                  │                              │ Subtotal/layer ≈ 13,760 │
    │                  │                              │ (vs 331,776 original)   │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ Total 9x9 params │ 4×1,327,104 + 4×331,776     │ 4×54,656 + 4×13,760    │
    │ (8 layers)       │ = 6,635,520                  │ = 273,664               │
    │                  │                              │ 95.9% reduction         │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ Inference(reparam)│ Same as above                │ DW 9x9 merged (no extra │
    │                  │                              │ branch overhead)+PW 1x1 │
    │                  │                              │ FLOPs also greatly reduced│
    └──────────────────────────────────────────────────────────────────────────┘

    Note: BN params in dilated branches are only present during training and
    merged into the DW conv weights at inference. The training param count
    includes these BN params; inference param count is slightly lower.
    """

    def __init__(self, in_channels, out_channels, mode="large", **kwargs):
        super(RepLKPAN, self).__init__()
        self.out_channels = out_channels
        self.is_repped = False
        weight_attr = paddle.nn.initializer.KaimingUniform()

        self.ins_conv = nn.LayerList()
        self.inp_conv = nn.LayerList()
        # pan head
        self.pan_head_conv = nn.LayerList()
        self.pan_lat_conv = nn.LayerList()

        for i in range(len(in_channels)):
            self.ins_conv.append(
                nn.Conv2D(
                    in_channels=in_channels[i],
                    out_channels=self.out_channels,
                    kernel_size=1,
                    weight_attr=ParamAttr(initializer=weight_attr),
                    bias_attr=False,
                )
            )

            self.inp_conv.append(
                DilatedReparamConv(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                )
            )

            if i > 0:
                self.pan_head_conv.append(
                    nn.Conv2D(
                        in_channels=self.out_channels // 4,
                        out_channels=self.out_channels // 4,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        weight_attr=ParamAttr(initializer=weight_attr),
                        bias_attr=False,
                    )
                )
            self.pan_lat_conv.append(
                DilatedReparamConv(
                    in_channels=self.out_channels // 4,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                )
            )

        self.intracl = False
        if "intracl" in kwargs.keys() and kwargs["intracl"] is True:
            self.intracl = kwargs["intracl"]
            self.incl1 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/4

        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)

        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)

        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)

        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        if self.training:
            return {"fuse": fuse, "aux_p4": out4, "aux_p3": out3, "aux_p2": out2}
        return fuse

    def rep(self):
        """Merge all DilatedReparamBlock branches and fuse PW+BN for deployment."""
        if self.is_repped:
            return
        for i in range(len(self.inp_conv)):
            self.inp_conv[i].rep()
        for i in range(len(self.pan_lat_conv)):
            self.pan_lat_conv[i].rep()
        self.is_repped = True


class ASFBlock(nn.Layer):
    """
    This code is referred from:
        https://github.com/MhLiao/DB/blob/master/decoders/feature_attention.py
    """

    def __init__(self, in_channels, inter_channels, out_features_num=4):
        """
        Adaptive Scale Fusion (ASF) block of DBNet++
        Args:
            in_channels: the number of channels in the input data
            inter_channels: the number of middle channels
            out_features_num: the number of fused stages
        """
        super(ASFBlock, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2D(in_channels, inter_channels, 3, padding=1)

        self.spatial_scale = nn.Sequential(
            # Nx1xHxW
            nn.Conv2D(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                bias_attr=False,
                padding=1,
                weight_attr=ParamAttr(initializer=weight_attr),
            ),
            nn.ReLU(),
            nn.Conv2D(
                in_channels=1,
                out_channels=1,
                kernel_size=1,
                bias_attr=False,
                weight_attr=ParamAttr(initializer=weight_attr),
            ),
            nn.Sigmoid(),
        )

        self.channel_scale = nn.Sequential(
            nn.Conv2D(
                in_channels=inter_channels,
                out_channels=out_features_num,
                kernel_size=1,
                bias_attr=False,
                weight_attr=ParamAttr(initializer=weight_attr),
            ),
            nn.Sigmoid(),
        )

    def forward(self, fuse_features, features_list):
        fuse_features = self.conv(fuse_features)
        spatial_x = paddle.mean(fuse_features, axis=1, keepdim=True)
        attention_scores = self.spatial_scale(spatial_x) + fuse_features
        attention_scores = self.channel_scale(attention_scores)
        assert len(features_list) == self.out_features_num

        out_list = []
        for i in range(self.out_features_num):
            out_list.append(attention_scores[:, i : i + 1] * features_list[i])
        return paddle.concat(out_list, axis=1)
