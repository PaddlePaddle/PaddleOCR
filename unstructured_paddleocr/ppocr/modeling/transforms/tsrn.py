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
"""
This code is refer from:
https://github.com/FudanVI/FudanOCR/blob/main/text-gestalt/model/tsrn.py
"""

import math
import paddle
import paddle.nn.functional as F
from paddle import nn
from collections import OrderedDict
import sys
import numpy as np
import warnings
import math, copy
import cv2

warnings.filterwarnings("ignore")

from .tps_spatial_transformer import TPSSpatialTransformer
from .stn import STN as STN_model
from ppocr.modeling.heads.sr_rensnet_transformer import Transformer


class TSRN(nn.Layer):
    def __init__(self,
                 in_channels,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=False,
                 hidden_units=32,
                 infer_mode=False,
                 **kwargs):
        super(TSRN, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2D(
                in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU())
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2),
                    RecurrentResidualBlock(2 * hidden_units))

        setattr(
            self,
            'block%d' % (srb_nums + 2),
            nn.Sequential(
                nn.Conv2D(
                    2 * hidden_units,
                    2 * hidden_units,
                    kernel_size=3,
                    padding=1),
                nn.BatchNorm2D(2 * hidden_units)))

        block_ = [
            UpsampleBLock(2 * hidden_units, 2)
            for _ in range(upsample_block_num)
        ]
        block_.append(
            nn.Conv2D(
                2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STN_model(
                in_channels=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')
        self.out_channels = in_channels

        self.r34_transformer = Transformer()
        for param in self.r34_transformer.parameters():
            param.trainable = False
        self.infer_mode = infer_mode

    def forward(self, x):
        output = {}
        if self.infer_mode:
            output["lr_img"] = x
            y = x
        else:
            output["lr_img"] = x[0]
            output["hr_img"] = x[1]
            y = x[0]
        if self.stn and self.training:
            _, ctrl_points_x = self.stn_head(y)
            y, _ = self.tps(y, ctrl_points_x)
        block = {'1': self.block1(y)}
        for i in range(self.srb_nums + 1):
            block[str(i + 2)] = getattr(self,
                                        'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))

        sr_img = paddle.tanh(block[str(self.srb_nums + 3)])

        output["sr_img"] = sr_img

        if self.training:
            hr_img = x[1]
            length = x[2]
            input_tensor = x[3]

            # add transformer 
            sr_pred, word_attention_map_pred, _ = self.r34_transformer(
                sr_img, length, input_tensor)

            hr_pred, word_attention_map_gt, _ = self.r34_transformer(
                hr_img, length, input_tensor)

            output["hr_img"] = hr_img
            output["hr_pred"] = hr_pred
            output["word_attention_map_gt"] = word_attention_map_gt
            output["sr_pred"] = sr_pred
            output["word_attention_map_pred"] = word_attention_map_pred

        return output


class RecurrentResidualBlock(nn.Layer):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2D(channels)
        self.gru1 = GruBlock(channels, channels)
        self.prelu = mish()
        self.conv2 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2D(channels)
        self.gru2 = GruBlock(channels, channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.gru1(residual.transpose([0, 1, 3, 2])).transpose(
            [0, 1, 3, 2])

        return self.gru2(x + residual)


class UpsampleBLock(nn.Layer):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2D(
            in_channels, in_channels * up_scale**2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class mish(nn.Layer):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (paddle.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2D(
            in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels,
                          out_channels // 2,
                          direction='bidirectional')

    def forward(self, x):
        # x: b, c, w, h
        x = self.conv1(x)
        x = x.transpose([0, 2, 3, 1])  # b, w, h, c
        batch_size, w, h, c = x.shape
        x = x.reshape([-1, h, c])  # b*w, h, c  
        x, _ = self.gru(x)
        x = x.reshape([-1, w, h, c])
        x = x.transpose([0, 3, 1, 2])
        return x
