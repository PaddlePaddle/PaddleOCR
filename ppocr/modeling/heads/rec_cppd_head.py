# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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

try:
    from collections import Callable
except:
    from collections.abc import Callable

import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F
from ppocr.modeling.heads.rec_nrtr_head import Embeddings
from ppocr.modeling.backbones.rec_svtrnet import (
    DropPath,
    Identity,
    trunc_normal_,
    zeros_,
    ones_,
    Mlp,
)


class Attention(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv):
        N, C = kv.shape[1:]
        QN = q.shape[1]
        q = (
            self.q(q)
            .reshape([-1, QN, self.num_heads, C // self.num_heads])
            .transpose([0, 2, 1, 3])
        )
        k, v = (
            self.kv(kv)
            .reshape([-1, N, 2, self.num_heads, C // self.num_heads])
            .transpose((2, 0, 3, 1, 4))
        )
        attn = q.matmul(k.transpose((0, 1, 3, 2))) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, QN, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EdgeDecoderLayer(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=[0.0, 0.0],
        act_layer=nn.GELU,
        norm_layer="nn.LayerNorm",
        epsilon=1e-6,
    ):
        super().__init__()

        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path[0]) if drop_path[0] > 0.0 else Identity()
        self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)

        self.p = nn.Linear(dim, dim)
        self.cv = nn.Linear(dim, dim)
        self.pv = nn.Linear(dim, dim)

        self.dim = dim
        self.num_heads = num_heads
        self.p_proj = nn.Linear(dim, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, p, cv, pv):
        pN = p.shape[1]
        vN = cv.shape[1]
        p_shortcut = p

        p1 = (
            self.p(p)
            .reshape([-1, pN, self.num_heads, self.dim // self.num_heads])
            .transpose([0, 2, 1, 3])
        )
        cv1 = (
            self.cv(cv)
            .reshape([-1, vN, self.num_heads, self.dim // self.num_heads])
            .transpose([0, 2, 1, 3])
        )
        pv1 = (
            self.pv(pv)
            .reshape([-1, vN, self.num_heads, self.dim // self.num_heads])
            .transpose([0, 2, 1, 3])
        )

        edge = F.softmax(p1.matmul(pv1.transpose((0, 1, 3, 2))), -1)  # B h N N
        p_c = (edge @ cv1).transpose((0, 2, 1, 3)).reshape((-1, pN, self.dim))

        x1 = self.norm1(p_shortcut + self.drop_path1(self.p_proj(p_c)))

        x = self.norm2(x1 + self.drop_path1(self.mlp(x1)))
        return x


class DecoderLayer(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer="nn.LayerNorm",
        epsilon=1e-6,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
            self.normkv = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm1 = norm_layer(dim)
            self.normkv = norm_layer(dim)
        else:
            raise TypeError("The norm_layer must be str or paddle.nn.LayerNorm class")
        self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm2 = norm_layer(dim)
        else:
            raise TypeError("The norm_layer must be str or paddle.nn.layer.Layer class")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, q, kv):
        x1 = self.norm1(q + self.drop_path(self.mixer(q, kv)))
        x = self.norm2(x1 + self.drop_path(self.mlp(x1)))
        return x


class CPPDHead(nn.Layer):
    def __init__(
        self,
        in_channels,
        dim,
        out_channels,
        num_layer=2,
        drop_path_rate=0.1,
        max_len=25,
        vis_seq=50,
        ch=False,
        **kwargs,
    ):
        super(CPPDHead, self).__init__()

        self.out_channels = out_channels  # none + 26 + 10
        self.dim = dim
        self.ch = ch
        self.max_len = max_len + 1  # max_len + eos
        self.char_node_embed = Embeddings(
            d_model=dim, vocab=self.out_channels, scale_embedding=True
        )
        self.pos_node_embed = Embeddings(
            d_model=dim, vocab=self.max_len, scale_embedding=True
        )
        dpr = np.linspace(0, drop_path_rate, num_layer + 1)

        self.char_node_decoder = nn.LayerList(
            [
                DecoderLayer(
                    dim=dim,
                    num_heads=dim // 32,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop_path=dpr[i],
                )
                for i in range(num_layer)
            ]
        )
        self.pos_node_decoder = nn.LayerList(
            [
                DecoderLayer(
                    dim=dim,
                    num_heads=dim // 32,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop_path=dpr[i],
                )
                for i in range(num_layer)
            ]
        )

        self.edge_decoder = EdgeDecoderLayer(
            dim=dim,
            num_heads=dim // 32,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path=dpr[num_layer : num_layer + 1],
        )

        self.char_pos_embed = self.create_parameter(
            shape=[1, self.max_len, dim], default_initializer=zeros_
        )
        self.add_parameter("char_pos_embed", self.char_pos_embed)
        self.vis_pos_embed = self.create_parameter(
            shape=[1, vis_seq, dim], default_initializer=zeros_
        )
        self.add_parameter("vis_pos_embed", self.vis_pos_embed)

        self.char_node_fc1 = nn.Linear(dim, max_len)
        self.pos_node_fc1 = nn.Linear(dim, self.max_len)

        self.edge_fc = nn.Linear(dim, self.out_channels)
        trunc_normal_(self.char_pos_embed)
        trunc_normal_(self.vis_pos_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x, targets=None, epoch=0):
        if self.training:
            return self.forward_train(x, targets, epoch)
        else:
            return self.forward_test(x)

    def forward_test(self, x):
        visual_feats = x + self.vis_pos_embed
        bs = visual_feats.shape[0]
        pos_node_embed = (
            self.pos_node_embed(paddle.arange(self.max_len)).unsqueeze(0)
            + self.char_pos_embed
        )
        pos_node_embed = paddle.tile(pos_node_embed, [bs, 1, 1])
        char_vis_node_query = visual_feats
        pos_vis_node_query = paddle.concat([pos_node_embed, visual_feats], 1)

        for char_decoder_layer, pos_decoder_layer in zip(
            self.char_node_decoder, self.pos_node_decoder
        ):
            char_vis_node_query = char_decoder_layer(
                char_vis_node_query, char_vis_node_query
            )
            pos_vis_node_query = pos_decoder_layer(
                pos_vis_node_query, pos_vis_node_query[:, self.max_len :, :]
            )
        pos_node_query = pos_vis_node_query[:, : self.max_len, :]
        char_vis_feats = char_vis_node_query

        pos_node_feats = self.edge_decoder(
            pos_node_query, char_vis_feats, char_vis_feats
        )  # B, 26, dim
        edge_feats = self.edge_fc(pos_node_feats)  # B, 26, 37
        edge_logits = F.softmax(edge_feats, -1)

        return edge_logits

    def forward_train(self, x, targets=None, epoch=0):
        visual_feats = x + self.vis_pos_embed
        bs = visual_feats.shape[0]

        if self.ch:
            char_node_embed = self.char_node_embed(targets[-2])
        else:
            char_node_embed = self.char_node_embed(
                paddle.arange(self.out_channels)
            ).unsqueeze(0)
            char_node_embed = paddle.tile(char_node_embed, [bs, 1, 1])
        counting_char_num = char_node_embed.shape[1]
        pos_node_embed = (
            self.pos_node_embed(paddle.arange(self.max_len)).unsqueeze(0)
            + self.char_pos_embed
        )
        pos_node_embed = paddle.tile(pos_node_embed, [bs, 1, 1])

        node_feats = []

        char_vis_node_query = paddle.concat([char_node_embed, visual_feats], 1)
        pos_vis_node_query = paddle.concat([pos_node_embed, visual_feats], 1)

        for char_decoder_layer, pos_decoder_layer in zip(
            self.char_node_decoder, self.pos_node_decoder
        ):
            char_vis_node_query = char_decoder_layer(
                char_vis_node_query, char_vis_node_query[:, counting_char_num:, :]
            )
            pos_vis_node_query = pos_decoder_layer(
                pos_vis_node_query, pos_vis_node_query[:, self.max_len :, :]
            )

        char_node_query = char_vis_node_query[:, :counting_char_num, :]
        pos_node_query = pos_vis_node_query[:, : self.max_len, :]

        char_vis_feats = char_vis_node_query[:, counting_char_num:, :]
        char_node_feats1 = self.char_node_fc1(char_node_query)

        pos_node_feats1 = self.pos_node_fc1(pos_node_query)
        diag_mask = (
            paddle.eye(pos_node_feats1.shape[1])
            .unsqueeze(0)
            .tile([pos_node_feats1.shape[0], 1, 1])
        )
        pos_node_feats1 = (pos_node_feats1 * diag_mask).sum(-1)

        node_feats.append(char_node_feats1)
        node_feats.append(pos_node_feats1)

        pos_node_feats = self.edge_decoder(
            pos_node_query, char_vis_feats, char_vis_feats
        )  # B, 26, dim
        edge_feats = self.edge_fc(pos_node_feats)  # B, 26, 37

        return node_feats, edge_feats
