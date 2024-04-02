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
https://github.com/hikopensource/DAVAR-Lab-OCR/blob/main/davarocr/davar_rcg/models/sequence_heads/counting_head.py
"""
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal, KaimingNormal

from .rec_att_head import AttentionLSTM

kaiming_init_ = KaimingNormal()
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


class CNTHead(nn.Layer):
    def __init__(self,
                 embed_size=512,
                 encode_length=26,
                 out_channels=38,
                 **kwargs):
        super(CNTHead, self).__init__()

        self.out_channels = out_channels

        self.Wv_fusion = nn.Linear(embed_size, embed_size, bias_attr=False)
        self.Prediction_visual = nn.Linear(encode_length * embed_size,
                                           self.out_channels)

    def forward(self, visual_feature):

        b, c, h, w = visual_feature.shape
        visual_feature = visual_feature.reshape([b, c, h * w]).transpose(
            [0, 2, 1])
        visual_feature_num = self.Wv_fusion(visual_feature)  # batch * 26 * 512
        b, n, c = visual_feature_num.shape
        # using visual feature directly calculate the text length
        visual_feature_num = visual_feature_num.reshape([b, n * c])
        prediction_visual = self.Prediction_visual(visual_feature_num)

        return prediction_visual


class RFLHead(nn.Layer):
    def __init__(self,
                 in_channels=512,
                 hidden_size=256,
                 batch_max_legnth=25,
                 out_channels=38,
                 use_cnt=True,
                 use_seq=True,
                 **kwargs):

        super(RFLHead, self).__init__()
        assert use_cnt or use_seq
        self.use_cnt = use_cnt
        self.use_seq = use_seq
        if self.use_cnt:
            self.cnt_head = CNTHead(
                embed_size=in_channels,
                encode_length=batch_max_legnth + 1,
                out_channels=out_channels,
                **kwargs)
        if self.use_seq:
            self.seq_head = AttentionLSTM(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_size=hidden_size,
                **kwargs)
        self.batch_max_legnth = batch_max_legnth
        self.num_class = out_channels
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            kaiming_init_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)

    def forward(self, x, targets=None):
        cnt_inputs, seq_inputs = x
        if self.use_cnt:
            cnt_outputs = self.cnt_head(cnt_inputs)
        else:
            cnt_outputs = None
        if self.use_seq:
            if self.training:
                seq_outputs = self.seq_head(seq_inputs, targets[0],
                                            self.batch_max_legnth)
            else:
                seq_outputs = self.seq_head(seq_inputs, None,
                                            self.batch_max_legnth)
            return cnt_outputs, seq_outputs
        else:
            return cnt_outputs
