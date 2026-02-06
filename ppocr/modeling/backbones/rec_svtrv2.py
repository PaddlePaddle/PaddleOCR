# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

trunc_normal_ = TruncatedNormal(std=0.02)
normal_ = Normal
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (paddle.shape(x)[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias_attr=False,
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
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()),
            bias_attr=bias_attr,
        )
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


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
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        qkv = (
            self.qkv(x)
            .reshape((0, -1, 3, self.num_heads, self.head_dim))
            .transpose((2, 0, 3, 1, 4))
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
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
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        epsilon=1e-6,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, epsilon=epsilon)
        self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = self.norm1(x + self.drop_path(self.mixer(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class ConvBlock(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        epsilon=1e-6,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, epsilon=epsilon)
        self.mixer = nn.Conv2D(
            dim,
            dim,
            5,
            1,
            2,
            groups=num_heads,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, epsilon=epsilon)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        C, H, W = x.shape[1:]
        x = x + self.drop_path(self.mixer(x))
        x = self.norm1(x.flatten(2).transpose([0, 2, 1]))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        x = x.transpose([0, 2, 1]).reshape([0, C, H, W])
        return x


class FlattenTranspose(nn.Layer):
    def forward(self, x):
        return x.flatten(2).transpose([0, 2, 1])


class SubSample2D(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=[2, 1],
    ):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, sz):
        # print(x.shape)
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose([0, 2, 1]))
        x = x.transpose([0, 2, 1]).reshape([0, C, H, W])
        return x, [H, W]


class SubSample1D(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=[2, 1],
    ):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, sz):
        C = x.shape[-1]
        x = x.transpose([0, 2, 1]).reshape([0, C, sz[0], sz[1]])
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose([0, 2, 1]))
        return x, [H, W]


class IdentitySize(nn.Layer):
    def forward(self, x, sz):
        return x, sz


class SVTRStage(nn.Layer):
    def __init__(
        self,
        dim=64,
        out_dim=256,
        depth=3,
        mixer=["Local"] * 3,
        sub_k=[2, 1],
        num_heads=2,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path=[0.1] * 3,
        norm_layer=nn.LayerNorm,
        act=nn.GELU,
        eps=1e-6,
        downsample=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim

        conv_block_num = sum([1 if mix == "Conv" else 0 for mix in mixer])
        blocks = []
        for i in range(depth):
            if mixer[i] == "Conv":
                blocks.append(
                    ConvBlock(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        act_layer=act,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer,
                        epsilon=eps,
                    )
                )
            else:
                blocks.append(
                    Block(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        act_layer=act,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer,
                        epsilon=eps,
                    )
                )
            if i == conv_block_num - 1 and mixer[-1] != "Conv":
                blocks.append(FlattenTranspose())
        self.blocks = nn.Sequential(*blocks)
        if downsample:
            if mixer[-1] == "Conv":
                self.downsample = SubSample2D(dim, out_dim, stride=sub_k)
            elif mixer[-1] == "Global":
                self.downsample = SubSample1D(dim, out_dim, stride=sub_k)
        else:
            self.downsample = IdentitySize()

    def forward(self, x, sz):
        x = self.blocks(x)
        x, sz = self.downsample(x, sz)
        return x, sz


class ADDPosEmbed(nn.Layer):
    def __init__(self, feat_max_size=[8, 32], embed_dim=768):
        super().__init__()
        pos_embed = paddle.zeros(
            [1, feat_max_size[0] * feat_max_size[1], embed_dim], dtype=paddle.float32
        )
        trunc_normal_(pos_embed)
        pos_embed = pos_embed.transpose([0, 2, 1]).reshape(
            [1, embed_dim, feat_max_size[0], feat_max_size[1]]
        )
        self.pos_embed = self.create_parameter(
            [1, embed_dim, feat_max_size[0], feat_max_size[1]]
        )
        self.add_parameter("pos_embed", self.pos_embed)
        self.pos_embed.set_value(pos_embed)

    def forward(self, x):
        sz = x.shape[2:]
        x = x + self.pos_embed[:, :, : sz[0], : sz[1]]
        return x


class POPatchEmbed(nn.Layer):
    """Image to Patch Embedding"""

    def __init__(
        self,
        in_channels=3,
        feat_max_size=[8, 32],
        embed_dim=768,
        use_pos_embed=False,
        flatten=False,
    ):
        super().__init__()
        patch_embed = [
            ConvBNLayer(
                in_channels=in_channels,
                out_channels=embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias_attr=None,
            ),
            ConvBNLayer(
                in_channels=embed_dim // 2,
                out_channels=embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
                bias_attr=None,
            ),
        ]
        if use_pos_embed:
            patch_embed.append(ADDPosEmbed(feat_max_size, embed_dim))
        if flatten:
            patch_embed.append(FlattenTranspose())
        self.patch_embed = nn.Sequential(*patch_embed)

    def forward(self, x):
        sz = x.shape[2:]
        x = self.patch_embed(x)
        return x, [sz[0] // 4, sz[1] // 4]


class LastStage(nn.Layer):
    def __init__(self, in_channels, out_channels, last_drop, out_char_num):
        super().__init__()
        self.last_conv = nn.Linear(in_channels, out_channels, bias_attr=False)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=last_drop, mode="downscale_in_infer")

    def forward(self, x, sz):
        x = x.reshape([0, sz[0], sz[1], x.shape[-1]])
        x = x.mean(1)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        return x, [1, sz[1]]


class OutPool(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, sz):
        C = x.shape[-1]
        x = x.transpose([0, 2, 1]).reshape([0, C, sz[0], sz[1]])
        x = nn.functional.avg_pool2d(x, [sz[0], 2])
        return x, [1, sz[1] // 2]


class Feat2D(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, sz):
        C = x.shape[-1]
        x = x.transpose([0, 2, 1]).reshape([0, C, sz[0], sz[1]])
        return x, sz


class SVTRv2(nn.Layer):
    def __init__(
        self,
        max_sz=[32, 128],
        in_channels=3,
        out_channels=192,
        out_char_num=25,
        depths=[3, 6, 3],
        dims=[64, 128, 256],
        mixer=[["Conv"] * 3, ["Conv"] * 3 + ["Global"] * 3, ["Global"] * 3],
        use_pos_embed=False,
        sub_k=[[1, 1], [2, 1], [1, 1]],
        num_heads=[2, 4, 8],
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        last_drop=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        act=nn.GELU,
        last_stage=False,
        eps=1e-6,
        use_pool=False,
        feat2d=False,
        **kwargs,
    ):
        super().__init__()
        num_stages = len(depths)
        self.num_features = dims[-1]

        feat_max_size = [max_sz[0] // 4, max_sz[1] // 4]
        self.pope = POPatchEmbed(
            in_channels=in_channels,
            feat_max_size=feat_max_size,
            embed_dim=dims[0],
            use_pos_embed=use_pos_embed,
            flatten=mixer[0][0] != "Conv",
        )

        dpr = np.linspace(0, drop_path_rate, sum(depths))  # stochastic depth decay rule

        self.stages = nn.LayerList()
        for i_stage in range(num_stages):
            stage = SVTRStage(
                dim=dims[i_stage],
                out_dim=dims[i_stage + 1] if i_stage < num_stages - 1 else 0,
                depth=depths[i_stage],
                mixer=mixer[i_stage],
                sub_k=sub_k[i_stage],
                num_heads=num_heads[i_stage],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]) : sum(depths[: i_stage + 1])],
                norm_layer=norm_layer,
                act=act,
                downsample=False if i_stage == num_stages - 1 else True,
                eps=eps,
            )
            self.stages.append(stage)

        self.out_channels = self.num_features
        self.last_stage = last_stage
        if last_stage:
            self.out_channels = out_channels
            self.stages.append(
                LastStage(self.num_features, out_channels, last_drop, out_char_num)
            )
        if use_pool:
            self.stages.append(OutPool())

        if feat2d:
            self.stages.append(Feat2D())
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
        x, sz = self.pope(x)
        for stage in self.stages:
            x, sz = stage(x, sz)
        return x
