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
# reference from : https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/kie/heads/sdmgr_head.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr


class SDMGRHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 num_chars=92,
                 visual_dim=16,
                 fusion_dim=1024,
                 node_input=32,
                 node_embed=256,
                 edge_input=5,
                 edge_embed=256,
                 num_gnn=2,
                 num_classes=26,
                 bidirectional=False):
        super().__init__()

        self.fusion = Block([visual_dim, node_embed], node_embed, fusion_dim)
        self.node_embed = nn.Embedding(num_chars, node_input, 0)
        hidden = node_embed // 2 if bidirectional else node_embed
        self.rnn = nn.LSTM(
            input_size=node_input, hidden_size=hidden, num_layers=1)
        self.edge_embed = nn.Linear(edge_input, edge_embed)
        self.gnn_layers = nn.LayerList(
            [GNNLayer(node_embed, edge_embed) for _ in range(num_gnn)])
        self.node_cls = nn.Linear(node_embed, num_classes)
        self.edge_cls = nn.Linear(edge_embed, 2)

    def forward(self, input, targets):
        relations, texts, x = input
        node_nums, char_nums = [], []
        for text in texts:
            node_nums.append(text.shape[0])
            char_nums.append(paddle.sum((text > -1).astype(int), axis=-1))

        max_num = max([char_num.max() for char_num in char_nums])
        all_nodes = paddle.concat([
            paddle.concat(
                [text, paddle.zeros(
                    (text.shape[0], max_num - text.shape[1]))], -1)
            for text in texts
        ])
        temp = paddle.clip(all_nodes, min=0).astype(int)
        embed_nodes = self.node_embed(temp)
        rnn_nodes, _ = self.rnn(embed_nodes)

        b, h, w = rnn_nodes.shape
        nodes = paddle.zeros([b, w])
        all_nums = paddle.concat(char_nums)
        valid = paddle.nonzero((all_nums > 0).astype(int))
        temp_all_nums = (
            paddle.gather(all_nums, valid) - 1).unsqueeze(-1).unsqueeze(-1)
        temp_all_nums = paddle.expand(temp_all_nums, [
            temp_all_nums.shape[0], temp_all_nums.shape[1], rnn_nodes.shape[-1]
        ])
        temp_all_nodes = paddle.gather(rnn_nodes, valid)
        N, C, A = temp_all_nodes.shape
        one_hot = F.one_hot(
            temp_all_nums[:, 0, :], num_classes=C).transpose([0, 2, 1])
        one_hot = paddle.multiply(
            temp_all_nodes, one_hot.astype("float32")).sum(axis=1, keepdim=True)
        t = one_hot.expand([N, 1, A]).squeeze(1)
        nodes = paddle.scatter(nodes, valid.squeeze(1), t)

        if x is not None:
            nodes = self.fusion([x, nodes])

        all_edges = paddle.concat(
            [rel.reshape([-1, rel.shape[-1]]) for rel in relations])
        embed_edges = self.edge_embed(all_edges.astype('float32'))
        embed_edges = F.normalize(embed_edges)

        for gnn_layer in self.gnn_layers:
            nodes, cat_nodes = gnn_layer(nodes, embed_edges, node_nums)

        node_cls, edge_cls = self.node_cls(nodes), self.edge_cls(cat_nodes)
        return node_cls, edge_cls


class GNNLayer(nn.Layer):
    def __init__(self, node_dim=256, edge_dim=256):
        super().__init__()
        self.in_fc = nn.Linear(node_dim * 2 + edge_dim, node_dim)
        self.coef_fc = nn.Linear(node_dim, 1)
        self.out_fc = nn.Linear(node_dim, node_dim)
        self.relu = nn.ReLU()

    def forward(self, nodes, edges, nums):
        start, cat_nodes = 0, []
        for num in nums:
            sample_nodes = nodes[start:start + num]
            cat_nodes.append(
                paddle.concat([
                    paddle.expand(sample_nodes.unsqueeze(1), [-1, num, -1]),
                    paddle.expand(sample_nodes.unsqueeze(0), [num, -1, -1])
                ], -1).reshape([num**2, -1]))
            start += num
        cat_nodes = paddle.concat([paddle.concat(cat_nodes), edges], -1)
        cat_nodes = self.relu(self.in_fc(cat_nodes))
        coefs = self.coef_fc(cat_nodes)

        start, residuals = 0, []
        for num in nums:
            residual = F.softmax(
                -paddle.eye(num).unsqueeze(-1) * 1e9 +
                coefs[start:start + num**2].reshape([num, num, -1]), 1)
            residuals.append((residual * cat_nodes[start:start + num**2]
                              .reshape([num, num, -1])).sum(1))
            start += num**2

        nodes += self.relu(self.out_fc(paddle.concat(residuals)))
        return [nodes, cat_nodes]


class Block(nn.Layer):
    def __init__(self,
                 input_dims,
                 output_dim,
                 mm_dim=1600,
                 chunks=20,
                 rank=15,
                 shared=False,
                 dropout_input=0.,
                 dropout_pre_lin=0.,
                 dropout_output=0.,
                 pos_norm='before_cat'):
        super().__init__()
        self.rank = rank
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert (pos_norm in ['before_cat', 'after_cat'])
        self.pos_norm = pos_norm
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = (self.linear0
                        if shared else nn.Linear(input_dims[1], mm_dim))
        self.merge_linears0 = nn.LayerList()
        self.merge_linears1 = nn.LayerList()
        self.chunks = self.chunk_sizes(mm_dim, chunks)
        for size in self.chunks:
            ml0 = nn.Linear(size, size * rank)
            self.merge_linears0.append(ml0)
            ml1 = ml0 if shared else nn.Linear(size, size * rank)
            self.merge_linears1.append(ml1)
        self.linear_out = nn.Linear(mm_dim, output_dim)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bs = x1.shape[0]
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = paddle.split(x0, self.chunks, -1)
        x1_chunks = paddle.split(x1, self.chunks, -1)
        zs = []
        for x0_c, x1_c, m0, m1 in zip(x0_chunks, x1_chunks, self.merge_linears0,
                                      self.merge_linears1):
            m = m0(x0_c) * m1(x1_c)  # bs x split_size*rank
            m = m.reshape([bs, self.rank, -1])
            z = paddle.sum(m, 1)
            if self.pos_norm == 'before_cat':
                z = paddle.sqrt(F.relu(z)) - paddle.sqrt(F.relu(-z))
                z = F.normalize(z)
            zs.append(z)
        z = paddle.concat(zs, 1)
        if self.pos_norm == 'after_cat':
            z = paddle.sqrt(F.relu(z)) - paddle.sqrt(F.relu(-z))
            z = F.normalize(z)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z

    def chunk_sizes(self, dim, chunks):
        split_size = (dim + chunks - 1) // chunks
        sizes_list = [split_size] * chunks
        sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)
        return sizes_list
