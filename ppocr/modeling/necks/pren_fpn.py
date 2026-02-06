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
https://github.com/RuijieJ/pren/blob/main/Nets/Aggregation.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F


class PoolAggregate(nn.Layer):
    def __init__(self, n_r, d_in, d_middle=None, d_out=None):
        super(PoolAggregate, self).__init__()
        if not d_middle:
            d_middle = d_in
        if not d_out:
            d_out = d_in

        self.d_in = d_in
        self.d_middle = d_middle
        self.d_out = d_out
        self.act = nn.Swish()

        self.n_r = n_r
        self.aggs = self._build_aggs()

    def _build_aggs(self):
        aggs = []
        for i in range(self.n_r):
            aggs.append(
                self.add_sublayer(
                    "{}".format(i),
                    nn.Sequential(
                        (
                            "conv1",
                            nn.Conv2D(
                                self.d_in, self.d_middle, 3, 2, 1, bias_attr=False
                            ),
                        ),
                        ("bn1", nn.BatchNorm(self.d_middle)),
                        ("act", self.act),
                        (
                            "conv2",
                            nn.Conv2D(
                                self.d_middle, self.d_out, 3, 2, 1, bias_attr=False
                            ),
                        ),
                        ("bn2", nn.BatchNorm(self.d_out)),
                    ),
                )
            )
        return aggs

    def forward(self, x):
        b = x.shape[0]
        outs = []
        for agg in self.aggs:
            y = agg(x)
            p = F.adaptive_avg_pool2d(y, 1)
            outs.append(p.reshape((b, 1, self.d_out)))
        out = paddle.concat(outs, 1)
        return out


class WeightAggregate(nn.Layer):
    def __init__(self, n_r, d_in, d_middle=None, d_out=None):
        super(WeightAggregate, self).__init__()
        if not d_middle:
            d_middle = d_in
        if not d_out:
            d_out = d_in

        self.n_r = n_r
        self.d_out = d_out
        self.act = nn.Swish()

        self.conv_n = nn.Sequential(
            ("conv1", nn.Conv2D(d_in, d_in, 3, 1, 1, bias_attr=False)),
            ("bn1", nn.BatchNorm(d_in)),
            ("act1", self.act),
            ("conv2", nn.Conv2D(d_in, n_r, 1, bias_attr=False)),
            ("bn2", nn.BatchNorm(n_r)),
            ("act2", nn.Sigmoid()),
        )
        self.conv_d = nn.Sequential(
            ("conv1", nn.Conv2D(d_in, d_middle, 3, 1, 1, bias_attr=False)),
            ("bn1", nn.BatchNorm(d_middle)),
            ("act1", self.act),
            ("conv2", nn.Conv2D(d_middle, d_out, 1, bias_attr=False)),
            ("bn2", nn.BatchNorm(d_out)),
        )

    def forward(self, x):
        b, _, h, w = x.shape

        hmaps = self.conv_n(x)
        fmaps = self.conv_d(x)
        r = paddle.bmm(
            hmaps.reshape((b, self.n_r, h * w)),
            fmaps.reshape((b, self.d_out, h * w)).transpose((0, 2, 1)),
        )
        return r


class GCN(nn.Layer):
    def __init__(self, d_in, n_in, d_out=None, n_out=None, dropout=0.1):
        super(GCN, self).__init__()
        if not d_out:
            d_out = d_in
        if not n_out:
            n_out = d_in

        self.conv_n = nn.Conv1D(n_in, n_out, 1)
        self.linear = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Swish()

    def forward(self, x):
        x = self.conv_n(x)
        x = self.dropout(self.linear(x))
        return self.act(x)


class PRENFPN(nn.Layer):
    def __init__(self, in_channels, n_r, d_model, max_len, dropout):
        super(PRENFPN, self).__init__()
        assert len(in_channels) == 3, "in_channels' length must be 3."
        c1, c2, c3 = in_channels  # the depths are from big to small
        # build fpn
        assert d_model % 3 == 0, "{} can't be divided by 3.".format(d_model)
        self.agg_p1 = PoolAggregate(n_r, c1, d_out=d_model // 3)
        self.agg_p2 = PoolAggregate(n_r, c2, d_out=d_model // 3)
        self.agg_p3 = PoolAggregate(n_r, c3, d_out=d_model // 3)

        self.agg_w1 = WeightAggregate(n_r, c1, 4 * c1, d_model // 3)
        self.agg_w2 = WeightAggregate(n_r, c2, 4 * c2, d_model // 3)
        self.agg_w3 = WeightAggregate(n_r, c3, 4 * c3, d_model // 3)

        self.gcn_pool = GCN(d_model, n_r, d_model, max_len, dropout)
        self.gcn_weight = GCN(d_model, n_r, d_model, max_len, dropout)

        self.out_channels = d_model

    def forward(self, inputs):
        f3, f5, f7 = inputs

        rp1 = self.agg_p1(f3)
        rp2 = self.agg_p2(f5)
        rp3 = self.agg_p3(f7)
        rp = paddle.concat([rp1, rp2, rp3], 2)  # [b,nr,d]

        rw1 = self.agg_w1(f3)
        rw2 = self.agg_w2(f5)
        rw3 = self.agg_w3(f7)
        rw = paddle.concat([rw1, rw2, rw3], 2)  # [b,nr,d]

        y1 = self.gcn_pool(rp)
        y2 = self.gcn_weight(rw)
        y = 0.5 * (y1 + y2)
        return y  # [b,max_len,d]
