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
from collections import namedtuple
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ['EfficientNetb3']


class EffB3Params:
    @staticmethod
    def get_global_params():
        """
        The fllowing are efficientnetb3's arch superparams, but to fit for scene 
        text recognition task, the resolution(image_size) here is changed 
        from 300 to 64.
        """
        GlobalParams = namedtuple('GlobalParams', [
            'drop_connect_rate', 'width_coefficient', 'depth_coefficient',
            'depth_divisor', 'image_size'
        ])
        global_params = GlobalParams(
            drop_connect_rate=0.3,
            width_coefficient=1.2,
            depth_coefficient=1.4,
            depth_divisor=8,
            image_size=64)
        return global_params

    @staticmethod
    def get_block_params():
        BlockParams = namedtuple('BlockParams', [
            'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
            'expand_ratio', 'id_skip', 'se_ratio', 'stride'
        ])
        block_params = [
            BlockParams(3, 1, 32, 16, 1, True, 0.25, 1),
            BlockParams(3, 2, 16, 24, 6, True, 0.25, 2),
            BlockParams(5, 2, 24, 40, 6, True, 0.25, 2),
            BlockParams(3, 3, 40, 80, 6, True, 0.25, 2),
            BlockParams(5, 3, 80, 112, 6, True, 0.25, 1),
            BlockParams(5, 4, 112, 192, 6, True, 0.25, 2),
            BlockParams(3, 1, 192, 320, 6, True, 0.25, 1)
        ]
        return block_params


class EffUtils:
    @staticmethod
    def round_filters(filters, global_params):
        """Calculate and round number of filters based on depth multiplier."""
        multiplier = global_params.width_coefficient
        if not multiplier:
            return filters
        divisor = global_params.depth_divisor
        filters *= multiplier
        new_filters = int(filters + divisor / 2) // divisor * divisor
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


class ConvBlock(nn.Layer):
    def __init__(self, block_params):
        super(ConvBlock, self).__init__()
        self.block_args = block_params
        self.has_se = (self.block_args.se_ratio is not None) and \
            (0 < self.block_args.se_ratio <= 1)
        self.id_skip = block_params.id_skip

        # expansion phase
        self.input_filters = self.block_args.input_filters
        output_filters = \
            self.block_args.input_filters * self.block_args.expand_ratio
        if self.block_args.expand_ratio != 1:
            self.expand_conv = nn.Conv2D(
                self.input_filters, output_filters, 1, bias_attr=False)
            self.bn0 = nn.BatchNorm(output_filters)

        # depthwise conv phase
        k = self.block_args.kernel_size
        s = self.block_args.stride
        self.depthwise_conv = nn.Conv2D(
            output_filters,
            output_filters,
            groups=output_filters,
            kernel_size=k,
            stride=s,
            padding='same',
            bias_attr=False)
        self.bn1 = nn.BatchNorm(output_filters)

        # squeeze and excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1,
                                        int(self.block_args.input_filters *
                                            self.block_args.se_ratio))
            self.se_reduce = nn.Conv2D(output_filters, num_squeezed_channels, 1)
            self.se_expand = nn.Conv2D(num_squeezed_channels, output_filters, 1)

        # output phase
        self.final_oup = self.block_args.output_filters
        self.project_conv = nn.Conv2D(
            output_filters, self.final_oup, 1, bias_attr=False)
        self.bn2 = nn.BatchNorm(self.final_oup)
        self.swish = nn.Swish()

    def drop_connect(self, inputs, p, training):
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
        if self.block_args.expand_ratio != 1:
            x = self.swish(self.bn0(self.expand_conv(inputs)))
        x = self.swish(self.bn1(self.depthwise_conv(x)))

        # squeeze and excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self.se_expand(self.swish(self.se_reduce(x_squeezed)))
            x = F.sigmoid(x_squeezed) * x
        x = self.bn2(self.project_conv(x))

        # skip conntection and drop connect
        if self.id_skip and self.block_args.stride == 1 and \
            self.input_filters == self.final_oup:
            if drop_connect_rate:
                x = self.drop_connect(
                    x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x


class EfficientNetb3_PREN(nn.Layer):
    def __init__(self, in_channels):
        super(EfficientNetb3_PREN, self).__init__()
        self.blocks_params = EffB3Params.get_block_params()
        self.global_params = EffB3Params.get_global_params()
        self.out_channels = []
        # stem
        stem_channels = EffUtils.round_filters(32, self.global_params)
        self.conv_stem = nn.Conv2D(
            in_channels, stem_channels, 3, 2, padding='same', bias_attr=False)
        self.bn0 = nn.BatchNorm(stem_channels)

        self.blocks = []
        # to extract three feature maps for fpn based on efficientnetb3 backbone
        self.concerned_block_idxes = [7, 17, 25]
        concerned_idx = 0
        for i, block_params in enumerate(self.blocks_params):
            block_params = block_params._replace(
                input_filters=EffUtils.round_filters(block_params.input_filters,
                                                     self.global_params),
                output_filters=EffUtils.round_filters(
                    block_params.output_filters, self.global_params),
                num_repeat=EffUtils.round_repeats(block_params.num_repeat,
                                                  self.global_params))
            self.blocks.append(
                self.add_sublayer("{}-0".format(i), ConvBlock(block_params)))
            concerned_idx += 1
            if concerned_idx in self.concerned_block_idxes:
                self.out_channels.append(block_params.output_filters)
            if block_params.num_repeat > 1:
                block_params = block_params._replace(
                    input_filters=block_params.output_filters, stride=1)
            for j in range(block_params.num_repeat - 1):
                self.blocks.append(
                    self.add_sublayer('{}-{}'.format(i, j + 1),
                                      ConvBlock(block_params)))
                concerned_idx += 1
                if concerned_idx in self.concerned_block_idxes:
                    self.out_channels.append(block_params.output_filters)

        self.swish = nn.Swish()

    def forward(self, inputs):
        outs = []
        
        x = self.swish(self.bn0(self.conv_stem(inputs)))
        for idx, block in enumerate(self.blocks):
            drop_connect_rate = self.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self.concerned_block_idxes:
                outs.append(x)
        return outs
