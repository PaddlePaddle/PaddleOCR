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

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)

        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name="bn_" + name + "_scale"),
            bias_attr=ParamAttr(name="bn_" + name + "_offset"),
            moving_mean_name="bn_" + name + "_mean",
            moving_variance_name="bn_" + name + "_variance")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SAST_Header1(nn.Layer):
    def __init__(self, in_channels, **kwargs):
        super(SAST_Header1, self).__init__()
        out_channels = [64, 64, 128]
        self.score_conv = nn.Sequential(
            ConvBNLayer(in_channels, out_channels[0], 1, 1, act='relu', name='f_score1'),
            ConvBNLayer(out_channels[0], out_channels[1], 3, 1, act='relu', name='f_score2'),
            ConvBNLayer(out_channels[1], out_channels[2], 1, 1, act='relu', name='f_score3'),
            ConvBNLayer(out_channels[2], 1, 3, 1, act=None, name='f_score4')
        )
        self.border_conv = nn.Sequential(
            ConvBNLayer(in_channels, out_channels[0], 1, 1, act='relu', name='f_border1'),
            ConvBNLayer(out_channels[0], out_channels[1], 3, 1, act='relu', name='f_border2'),
            ConvBNLayer(out_channels[1], out_channels[2], 1, 1, act='relu', name='f_border3'),
            ConvBNLayer(out_channels[2], 4, 3, 1, act=None, name='f_border4')            
        )

    def forward(self, x):
        f_score = self.score_conv(x)
        f_score = F.sigmoid(f_score)
        f_border = self.border_conv(x)
        return f_score, f_border


class SAST_Header2(nn.Layer):
    def __init__(self, in_channels, **kwargs):
        super(SAST_Header2, self).__init__()
        out_channels = [64, 64, 128]
        self.tvo_conv = nn.Sequential(
            ConvBNLayer(in_channels, out_channels[0], 1, 1, act='relu', name='f_tvo1'),
            ConvBNLayer(out_channels[0], out_channels[1], 3, 1, act='relu', name='f_tvo2'),
            ConvBNLayer(out_channels[1], out_channels[2], 1, 1, act='relu', name='f_tvo3'),
            ConvBNLayer(out_channels[2], 8, 3, 1, act=None, name='f_tvo4')
        )
        self.tco_conv = nn.Sequential(
            ConvBNLayer(in_channels, out_channels[0], 1, 1, act='relu', name='f_tco1'),
            ConvBNLayer(out_channels[0], out_channels[1], 3, 1, act='relu', name='f_tco2'),
            ConvBNLayer(out_channels[1], out_channels[2], 1, 1, act='relu', name='f_tco3'),
            ConvBNLayer(out_channels[2], 2, 3, 1, act=None, name='f_tco4')            
        )

    def forward(self, x):
        f_tvo = self.tvo_conv(x)
        f_tco = self.tco_conv(x)
        return f_tvo, f_tco


class SASTHead(nn.Layer):
    """
    """
    def __init__(self, in_channels, **kwargs):
        super(SASTHead, self).__init__()

        self.head1 = SAST_Header1(in_channels)
        self.head2 = SAST_Header2(in_channels)

    def forward(self, x):
        f_score, f_border = self.head1(x)
        f_tvo, f_tco = self.head2(x)

        predicts = {}
        predicts['f_score'] = f_score
        predicts['f_border'] = f_border
        predicts['f_tvo'] = f_tvo
        predicts['f_tco'] = f_tco
        return predicts