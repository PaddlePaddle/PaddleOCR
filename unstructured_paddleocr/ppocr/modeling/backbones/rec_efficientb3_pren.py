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
"""
Code is refer from:
https://github.com/RuijieJ/pren/blob/main/Nets/EfficientNet.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import re
import collections
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ["EfficientNetb3_PREN"]

GlobalParams = collections.namedtuple(
    "GlobalParams",
    [
        "batch_norm_momentum",
        "batch_norm_epsilon",
        "dropout_rate",
        "num_classes",
        "width_coefficient",
        "depth_coefficient",
        "depth_divisor",
        "min_depth",
        "drop_connect_rate",
        "image_size",
    ],
)

BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "kernel_size",
        "num_repeat",
        "input_filters",
        "output_filters",
        "expand_ratio",
        "id_skip",
        "stride",
        "se_ratio",
    ],
)


class BlockDecoder:
    @staticmethod
    def _decode_block_string(block_string):
        assert isinstance(block_string, str)

        ops = block_string.split("_")
        options = {}
        for op in ops:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        assert ("s" in options and len(options["s"]) == 1) or (
            len(options["s"]) == 2 and options["s"][0] == options["s"][1]
        )

        return BlockArgs(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=[int(options["s"][0])],
        )

    @staticmethod
    def decode(string_list):
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args


def efficientnet(
    width_coefficient=None,
    depth_coefficient=None,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    image_size=None,
    num_classes=1000,
):
    blocks_args = [
        "r1_k3_s11_e1_i32_o16_se0.25",
        "r2_k3_s22_e6_i16_o24_se0.25",
        "r2_k5_s22_e6_i24_o40_se0.25",
        "r3_k3_s22_e6_i40_o80_se0.25",
        "r3_k5_s11_e6_i80_o112_se0.25",
        "r4_k5_s22_e6_i112_o192_se0.25",
        "r1_k3_s11_e6_i192_o320_se0.25",
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )
    return blocks_args, global_params


class EffUtils:
    @staticmethod
    def round_filters(filters, global_params):
        """Calculate and round number of filters based on depth multiplier."""
        multiplier = global_params.width_coefficient
        if not multiplier:
            return filters
        divisor = global_params.depth_divisor
        min_depth = global_params.min_depth
        filters *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    @staticmethod
    def round_repeats(repeats, global_params):
        """Round number of filters based on depth multiplier."""
        multiplier = global_params.depth_coefficient
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))


class MbConvBlock(nn.Layer):
    def __init__(self, block_args):
        super(MbConvBlock, self).__init__()
        self._block_args = block_args
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1
        )
        self.id_skip = block_args.id_skip

        # expansion phase
        self.inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = nn.Conv2D(self.inp, oup, 1, bias_attr=False)
            self._bn0 = nn.BatchNorm(oup)

        # depthwise conv phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        if isinstance(s, list):
            s = s[0]
        self._depthwise_conv = nn.Conv2D(
            oup,
            oup,
            groups=oup,
            kernel_size=k,
            stride=s,
            padding="same",
            bias_attr=False,
        )
        self._bn1 = nn.BatchNorm(oup)

        # squeeze and excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se_reduce = nn.Conv2D(oup, num_squeezed_channels, 1)
            self._se_expand = nn.Conv2D(num_squeezed_channels, oup, 1)

        # output phase and some util class
        self.final_oup = self._block_args.output_filters
        self._project_conv = nn.Conv2D(oup, self.final_oup, 1, bias_attr=False)
        self._bn2 = nn.BatchNorm(self.final_oup)
        self._swish = nn.Swish()

    def _drop_connect(self, inputs, p, training):
        if not training:
            return inputs
        batch_size = inputs.shape[0]
        keep_prob = 1 - p
        random_tensor = keep_prob
        random_tensor += paddle.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)
        random_tensor = paddle.to_tensor(random_tensor, place=inputs.place)
        binary_tensor = paddle.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output

    def forward(self, inputs, drop_connect_rate=None):
        # expansion and depthwise conv
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # squeeze and excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = F.sigmoid(x_squeezed) * x
        x = self._bn2(self._project_conv(x))

        # skip conntection and drop connect
        if self.id_skip and self._block_args.stride == 1 and self.inp == self.final_oup:
            if drop_connect_rate:
                x = self._drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x


class EfficientNetb3_PREN(nn.Layer):
    def __init__(self, in_channels):
        super(EfficientNetb3_PREN, self).__init__()
        """
        the fllowing are efficientnetb3's superparams,
        they means efficientnetb3 network's width, depth, resolution and
        dropout respectively, to fit for text recognition task, the resolution
        here is changed from 300 to 64.
        """
        w, d, s, p = 1.2, 1.4, 64, 0.3
        self._blocks_args, self._global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s
        )
        self.out_channels = []
        # stem
        out_channels = EffUtils.round_filters(32, self._global_params)
        self._conv_stem = nn.Conv2D(
            in_channels, out_channels, 3, 2, padding="same", bias_attr=False
        )
        self._bn0 = nn.BatchNorm(out_channels)

        # build blocks
        self._blocks = []
        # to extract three feature maps for fpn based on efficientnetb3 backbone
        self._concerned_block_idxes = [7, 17, 25]
        _concerned_idx = 0
        for i, block_args in enumerate(self._blocks_args):
            block_args = block_args._replace(
                input_filters=EffUtils.round_filters(
                    block_args.input_filters, self._global_params
                ),
                output_filters=EffUtils.round_filters(
                    block_args.output_filters, self._global_params
                ),
                num_repeat=EffUtils.round_repeats(
                    block_args.num_repeat, self._global_params
                ),
            )
            self._blocks.append(self.add_sublayer(f"{i}-0", MbConvBlock(block_args)))
            _concerned_idx += 1
            if _concerned_idx in self._concerned_block_idxes:
                self.out_channels.append(block_args.output_filters)
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for j in range(block_args.num_repeat - 1):
                self._blocks.append(
                    self.add_sublayer(f"{i}-{j+1}", MbConvBlock(block_args))
                )
                _concerned_idx += 1
                if _concerned_idx in self._concerned_block_idxes:
                    self.out_channels.append(block_args.output_filters)

        self._swish = nn.Swish()

    def forward(self, inputs):
        outs = []
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self._concerned_block_idxes:
                outs.append(x)
        return outs
