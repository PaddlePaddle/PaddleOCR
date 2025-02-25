# copyright (c) 2025 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import paddle
import paddle.nn as nn
from ppocr.modeling.backbones.rec_svtrnet import DropPath, Identity, Mlp
from paddle import ParamAttr
from paddle.nn.initializer import (
    TruncatedNormal,
    Constant,
    Normal,
    KaimingNormal,
    KaimingUniform,
)

trunc_normal_ = TruncatedNormal(std=0.02)
normal_ = Normal
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
        act=nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingUniform()),
            bias_attr=bias,
        )
        self.norm = nn.BatchNorm2D(num_features=out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class ConvMixer(nn.Layer):
    def __init__(self, dim, num_heads=8, HW=[8, 25], local_k=[3, 3]):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2D(
            in_channels=dim,
            out_channels=dim,
            kernel_size=local_k,
            stride=1,
            padding=[local_k[0] // 2, local_k[1] // 2],
            groups=num_heads,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
        )

    def forward(self, x, w):
        x = x.transpose(perm=dim2perm(x.ndim, 1, 2)).reshape(
            [tuple(x.shape)[0], self.dim, -1, w]
        )
        x = self.local_mixer(x)
        x = x.flatten(start_axis=2).transpose(
            perm=dim2perm(x.flatten(start_axis=2).ndim, 1, 2)
        )
        return x


class ConvMlp(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        groups=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(
            in_channels=in_features,
            out_channels=hidden_features,
            kernel_size=1,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
        )
        self.act = act_layer()
        self.fc2 = nn.Conv2D(
            in_channels=hidden_features, out_channels=out_features, kernel_size=1
        )
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvBlock(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mixer="Global",
        local_mixer=[7, 11],
        HW=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer="nn.LayerNorm",
        eps=1e-06,
        prenorm=True,
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2D(num_features=dim)
        self.local_mixer = nn.Conv2D(
            in_channels=dim,
            out_channels=dim,
            kernel_size=[5, 5],
            stride=1,
            padding=[2, 2],
            groups=num_heads,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = nn.BatchNorm2D(num_features=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.prenorm = prenorm

    def forward(self, x):
        x = self.norm1(x + self.drop_path(self.local_mixer(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class Attention(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        mixer="Global",
        HW=None,
        local_k=[7, 11],
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == "Local" and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = paddle.ones([H * W, H + hk - 1, W + wk - 1], dtype="float32")
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h : h + hk, w : w + wk] = 0.0
            mask_paddle = mask[:, hk // 2 : H + hk // 2, wk // 2 : W + wk // 2].flatten(
                1
            )
            mask_inf = paddle.full([H * W, H * W], "-inf", dtype="float32")
            mask = paddle.where(mask_paddle < 1, mask_paddle, mask_inf)
            self.register_buffer("mask", mask.unsqueeze([0, 1]))
        self.mixer = mixer

    def forward(self, x):
        qkv = (
            self.qkv(x)
            .reshape((0, -1, 3, self.num_heads, self.head_dim))
            .transpose((2, 0, 3, 1, 4))
        )
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q.matmul(k.transpose((0, 1, 3, 2)))
        if self.mixer == "Local":
            attn += self.mask
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((0, -1, self.dim))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mixer="Global",
        local_mixer=[7, 11],
        HW=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer="nn.LayerNorm",
        eps=1e-6,
        prenorm=True,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=eps)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == "Global" or mixer == "Local":
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif mixer == "Conv":
            self.mixer = ConvMixer(dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=eps)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.prenorm = prenorm

    def forward(self, x, w):
        x = self.norm1(x + self.drop_path(self.mixer(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x, w


class PatchEmbed(nn.Layer):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size=[32, 100],
        in_channels=3,
        embed_dim=768,
        sub_num=2,
        patch_size=[4, 4],
        mode="pope",
    ):
        super().__init__()
        num_patches = img_size[1] // 2**sub_num * (img_size[0] // 2**sub_num)
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if mode == "pope":
            if sub_num == 2:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias=False,
                    ),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias=False,
                    ),
                )
            if sub_num == 3:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias=False,
                    ),
                    ConvBNLayer(
                        in_channels=embed_dim // 4,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias=False,
                    ),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias=False,
                    ),
                )
        elif mode == "linear":
            self.proj = nn.Conv2D(
                in_channels=1,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
            self.num_patches = (
                img_size[0] // patch_size[0] * img_size[1] // patch_size[1]
            )

    def forward(self, x):
        x = self.proj(x)
        return x


class SubSample(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        types="Pool",
        stride=[2, 1],
        sub_norm="nn.LayerNorm",
        act=None,
    ):
        super().__init__()
        self.types = types
        if types == "Pool":
            self.avgpool = nn.AvgPool2D(
                kernel_size=[3, 5], stride=stride, padding=[1, 2], exclusive=False
            )
            self.maxpool = nn.MaxPool2D(
                kernel_size=[3, 5], stride=stride, padding=[1, 2]
            )
            self.proj = nn.Linear(in_features=in_channels, out_features=out_channels)
        else:
            self.conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            )
        self.dim = in_channels
        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x, w):
        if self.types == "Pool":
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(
                x.flatten(start_axis=2).transpose(
                    perm=dim2perm(x.flatten(start_axis=2).ndim, 1, 2)
                )
            )
        else:
            x = x.transpose(perm=dim2perm(x.ndim, 1, 2)).reshape(
                [tuple(x.shape)[0], self.dim, -1, w]
            )
            x = self.conv(x)
            out = x.flatten(start_axis=2).transpose(
                perm=dim2perm(x.flatten(start_axis=2).ndim, 1, 2)
            )
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out, w


class FlattenTranspose(nn.Layer):
    def forward(self, x):
        return x.flatten(start_axis=2).transpose(
            perm=dim2perm(x.flatten(start_axis=2).ndim, 1, 2)
        )


class DownSConv(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=[2, 1],
            padding=1,
        )
        self.norm = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, x, w):
        B, N, C = tuple(x.shape)
        x = x.transpose(perm=dim2perm(x.ndim, 1, 2)).reshape([B, C, -1, w])
        x = self.conv(x)
        w = tuple(x.shape)[-1]
        x = self.norm(
            x.flatten(start_axis=2).transpose(
                perm=dim2perm(x.flatten(start_axis=2).ndim, 1, 2)
            )
        )
        return x, w


class SVTRNet2DPos(nn.Layer):
    def __init__(
        self,
        img_size=[32, -1],
        in_channels=3,
        embed_dim=[64, 128, 256],
        depth=[3, 6, 3],
        num_heads=[2, 4, 8],
        mixer=["Local"] * 6 + ["Global"] * 6,
        local_mixer=[[7, 11], [7, 11], [7, 11]],
        patch_merging="Conv",
        pool_size=[2, 1],
        max_size=[16, 32],
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        last_drop=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer="nn.LayerNorm",
        eps=1e-06,
        act="nn.GELU",
        last_stage=True,
        sub_num=2,
        use_first_sub=True,
        flatten=False,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.flatten = flatten
        patch_merging = (
            None
            if patch_merging != "Conv" and patch_merging != "Pool"
            else patch_merging
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            sub_num=sub_num,
        )
        if img_size[1] == -1:
            self.HW = [img_size[0] // 2**sub_num, -1]
        else:
            self.HW = [img_size[0] // 2**sub_num, img_size[1] // 2**sub_num]
        pos_embed = paddle.zeros(
            shape=[1, max_size[0] * max_size[1], embed_dim[0]], dtype="float32"
        )
        init_TruncatedNormal = nn.initializer.TruncatedNormal(mean=0, std=0.02)
        init_TruncatedNormal(pos_embed)
        self.pos_embed = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=pos_embed.transpose(perm=dim2perm(pos_embed.ndim, 1, 2)).reshape(
                [1, embed_dim[0], max_size[0], max_size[1]]
            ),
            trainable=True,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        conv_block_num = sum(
            [(1 if mixer_type == "ConvB" else 0) for mixer_type in mixer]
        )
        Block_unit = [ConvBlock for _ in range(conv_block_num)] + [
            Block for _ in range(len(mixer) - conv_block_num)
        ]
        HW = self.HW
        dpr = np.linspace(0, drop_path_rate, sum(depth))
        self.conv_blocks1 = nn.LayerList(
            sublayers=[
                Block_unit[0 : depth[0]][i](
                    dim=embed_dim[0],
                    num_heads=num_heads[0],
                    mixer=mixer[0 : depth[0]][i],
                    HW=self.HW,
                    local_mixer=local_mixer[0],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=eval(act),
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[0 : depth[0]][i],
                    norm_layer=norm_layer,
                    eps=eps,
                )
                for i in range(depth[0])
            ]
        )
        if patch_merging is not None:
            if use_first_sub:
                stride = [2, 1]
                HW = [self.HW[0] // 2, self.HW[1]]
            else:
                stride = [1, 1]
                HW = self.HW
            sub_sample1 = nn.Sequential(
                nn.Conv2D(
                    in_channels=embed_dim[0],
                    out_channels=embed_dim[1],
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                ),
                nn.BatchNorm2D(num_features=embed_dim[1]),
            )
            self.conv_blocks1.append(sub_sample1)
        self.patch_merging = patch_merging
        self.trans_blocks = nn.LayerList()
        for i in range(depth[1]):
            block = Block_unit[depth[0] : depth[0] + depth[1]][i](
                dim=embed_dim[1],
                num_heads=num_heads[1],
                mixer=mixer[depth[0] : depth[0] + depth[1]][i],
                HW=HW,
                local_mixer=local_mixer[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0] : depth[0] + depth[1]][i],
                norm_layer=norm_layer,
                eps=eps,
            )
            if i + depth[0] < conv_block_num:
                self.conv_blocks1.append(block)
            else:
                self.trans_blocks.append(block)
        if patch_merging is not None:
            self.trans_blocks.append(DownSConv(embed_dim[1], embed_dim[2]))
            HW = [HW[0] // 2, -1]
        for i in range(depth[2]):
            self.trans_blocks.append(
                Block_unit[depth[0] + depth[1] :][i](
                    dim=embed_dim[2],
                    num_heads=num_heads[2],
                    mixer=mixer[depth[0] + depth[1] :][i],
                    HW=HW,
                    local_mixer=local_mixer[2],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=eval(act),
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[depth[0] + depth[1] :][i],
                    norm_layer=norm_layer,
                    eps=eps,
                )
            )
        self.last_stage = last_stage
        self.out_channels = embed_dim[-1]
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x):
        x = self.patch_embed(x)
        w = tuple(x.shape)[-1]
        x = x + self.pos_embed[:, :, : tuple(x.shape)[-2], :w]
        for blk in self.conv_blocks1:
            x = blk(x)
        x = x.flatten(start_axis=2).transpose(
            perm=dim2perm(x.flatten(start_axis=2).ndim, 1, 2)
        )
        for blk in self.trans_blocks:
            x, w = blk(x, w)
        B, N, C = tuple(x.shape)
        if not self.flatten:
            x = x.transpose(perm=dim2perm(x.ndim, 1, 2)).reshape([B, C, -1, w])
        return x
