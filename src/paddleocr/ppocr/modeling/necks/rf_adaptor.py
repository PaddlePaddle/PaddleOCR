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
This code is refer from:
https://github.com/hikopensource/DAVAR-Lab-OCR/blob/main/davarocr/davar_rcg/models/connects/single_block/RFAdaptor.py
"""

import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal, KaimingNormal

kaiming_init_ = KaimingNormal()
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


class S2VAdaptor(nn.Layer):
    """Semantic to Visual adaptation module"""

    def __init__(self, in_channels=512):
        super(S2VAdaptor, self).__init__()

        self.in_channels = in_channels  # 512

        # feature strengthen module, channel attention
        self.channel_inter = nn.Linear(
            self.in_channels, self.in_channels, bias_attr=False
        )
        self.channel_bn = nn.BatchNorm1D(self.in_channels)
        self.channel_act = nn.ReLU()
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            kaiming_init_(m.weight)
            if isinstance(m, nn.Conv2D) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm, nn.BatchNorm2D, nn.BatchNorm1D)):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, semantic):
        semantic_source = semantic  # batch, channel, height, width

        # feature transformation
        semantic = semantic.squeeze(2).transpose([0, 2, 1])  # batch, width, channel
        channel_att = self.channel_inter(semantic)  # batch, width, channel
        channel_att = channel_att.transpose([0, 2, 1])  # batch, channel, width
        channel_bn = self.channel_bn(channel_att)  # batch, channel, width
        channel_att = self.channel_act(channel_bn)  # batch, channel, width

        # Feature enhancement
        channel_output = semantic_source * channel_att.unsqueeze(
            -2
        )  # batch, channel, 1, width

        return channel_output


class V2SAdaptor(nn.Layer):
    """Visual to Semantic adaptation module"""

    def __init__(self, in_channels=512, return_mask=False):
        super(V2SAdaptor, self).__init__()

        # parameter initialization
        self.in_channels = in_channels
        self.return_mask = return_mask

        # output transformation
        self.channel_inter = nn.Linear(
            self.in_channels, self.in_channels, bias_attr=False
        )
        self.channel_bn = nn.BatchNorm1D(self.in_channels)
        self.channel_act = nn.ReLU()

    def forward(self, visual):
        # Feature enhancement
        visual = visual.squeeze(2).transpose([0, 2, 1])  # batch, width, channel
        channel_att = self.channel_inter(visual)  # batch, width, channel
        channel_att = channel_att.transpose([0, 2, 1])  # batch, channel, width
        channel_bn = self.channel_bn(channel_att)  # batch, channel, width
        channel_att = self.channel_act(channel_bn)  # batch, channel, width

        # size alignment
        channel_output = channel_att.unsqueeze(-2)  # batch, width, channel

        if self.return_mask:
            return channel_output, channel_att
        return channel_output


class RFAdaptor(nn.Layer):
    def __init__(self, in_channels=512, use_v2s=True, use_s2v=True, **kwargs):
        super(RFAdaptor, self).__init__()
        if use_v2s is True:
            self.neck_v2s = V2SAdaptor(in_channels=in_channels, **kwargs)
        else:
            self.neck_v2s = None
        if use_s2v is True:
            self.neck_s2v = S2VAdaptor(in_channels=in_channels, **kwargs)
        else:
            self.neck_s2v = None
        self.out_channels = in_channels

    def forward(self, x):
        visual_feature, rcg_feature = x
        if visual_feature is not None:
            (
                batch,
                source_channels,
                v_source_height,
                v_source_width,
            ) = visual_feature.shape
            visual_feature = visual_feature.reshape(
                [batch, source_channels, 1, v_source_height * v_source_width]
            )

        if self.neck_v2s is not None:
            v_rcg_feature = rcg_feature * self.neck_v2s(visual_feature)
        else:
            v_rcg_feature = rcg_feature

        if self.neck_s2v is not None:
            v_visual_feature = visual_feature + self.neck_s2v(rcg_feature)
        else:
            v_visual_feature = visual_feature
        if v_rcg_feature is not None:
            batch, source_channels, source_height, source_width = v_rcg_feature.shape
            v_rcg_feature = v_rcg_feature.reshape(
                [batch, source_channels, 1, source_height * source_width]
            )

            v_rcg_feature = v_rcg_feature.squeeze(2).transpose([0, 2, 1])
        return v_visual_feature, v_rcg_feature
