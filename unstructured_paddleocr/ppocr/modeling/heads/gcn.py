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
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/modules/gcn.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BatchNorm1D(nn.BatchNorm1D):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        momentum = 1 - momentum
        weight_attr = None
        bias_attr = None
        if not affine:
            weight_attr = paddle.ParamAttr(learning_rate=0.0)
            bias_attr = paddle.ParamAttr(learning_rate=0.0)
        super().__init__(
            num_features,
            momentum=momentum,
            epsilon=eps,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            use_global_stats=track_running_stats)


class MeanAggregator(nn.Layer):
    def forward(self, features, A):
        x = paddle.bmm(A, features)
        return x


class GraphConv(nn.Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = self.create_parameter(
            [in_dim * 2, out_dim],
            default_initializer=nn.initializer.XavierUniform())
        self.bias = self.create_parameter(
            [out_dim],
            is_bias=True,
            default_initializer=nn.initializer.Assign([0] * out_dim))

        self.aggregator = MeanAggregator()

    def forward(self, features, A):
        b, n, d = features.shape
        assert d == self.in_dim
        agg_feats = self.aggregator(features, A)
        cat_feats = paddle.concat([features, agg_feats], axis=2)
        out = paddle.einsum('bnd,df->bnf', cat_feats, self.weight)
        out = F.relu(out + self.bias)
        return out


class GCN(nn.Layer):
    def __init__(self, feat_len):
        super(GCN, self).__init__()
        self.bn0 = BatchNorm1D(feat_len, affine=False)
        self.conv1 = GraphConv(feat_len, 512)
        self.conv2 = GraphConv(512, 256)
        self.conv3 = GraphConv(256, 128)
        self.conv4 = GraphConv(128, 64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.PReLU(32), nn.Linear(32, 2))

    def forward(self, x, A, knn_inds):

        num_local_graphs, num_max_nodes, feat_len = x.shape

        x = x.reshape([-1, feat_len])
        x = self.bn0(x)
        x = x.reshape([num_local_graphs, num_max_nodes, feat_len])

        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)
        k = knn_inds.shape[-1]
        mid_feat_len = x.shape[-1]
        edge_feat = paddle.zeros([num_local_graphs, k, mid_feat_len])
        for graph_ind in range(num_local_graphs):
            edge_feat[graph_ind, :, :] = x[graph_ind][paddle.to_tensor(knn_inds[
                graph_ind])]
        edge_feat = edge_feat.reshape([-1, mid_feat_len])
        pred = self.classifier(edge_feat)

        return pred
