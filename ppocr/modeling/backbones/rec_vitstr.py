# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
https://github.com/roatienza/deep-text-recognition-benchmark/blob/master/modules/vitstr.py
"""

import numpy as np
import paddle
import paddle.nn as nn
from ppocr.modeling.backbones.rec_svtrnet import (
    Block,
    PatchEmbed,
    zeros_,
    trunc_normal_,
    ones_,
)

scale_dim_heads = {"tiny": [192, 3], "small": [384, 6], "base": [768, 12]}


class ViTSTR(nn.Layer):
    def __init__(
        self,
        img_size=[224, 224],
        in_channels=1,
        scale="tiny",
        seqlen=27,
        patch_size=[16, 16],
        embed_dim=None,
        depth=12,
        num_heads=None,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_path_rate=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer="nn.LayerNorm",
        act_layer="nn.GELU",
        epsilon=1e-6,
        out_channels=None,
        **kwargs,
    ):
        super().__init__()
        self.seqlen = seqlen
        embed_dim = embed_dim if embed_dim is not None else scale_dim_heads[scale][0]
        num_heads = num_heads if num_heads is not None else scale_dim_heads[scale][1]
        out_channels = out_channels if out_channels is not None else embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            mode="linear",
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = self.create_parameter(
            shape=[1, num_patches + 1, embed_dim], default_initializer=zeros_
        )
        self.add_parameter("pos_embed", self.pos_embed)
        self.cls_token = self.create_parameter(
            shape=[1, 1, embed_dim], default_initializer=zeros_
        )
        self.add_parameter("cls_token", self.cls_token)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=eval(act_layer),
                    epsilon=epsilon,
                    prenorm=False,
                )
                for i in range(depth)
            ]
        )
        self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)

        self.out_channels = out_channels

        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = paddle.tile(self.cls_token, repeat_times=[B, 1, 1])
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x[:, : self.seqlen]
        return x.transpose([0, 2, 1]).unsqueeze(2)
