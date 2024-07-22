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

"""
This code is refer from:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer_hybrid.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import repeat
import collections
import math
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppocr.modeling.backbones.rec_resnetv2 import (
    ResNetV2,
    StdConv2dSame,
    DropPath,
    get_padding,
)
from paddle.nn.initializer import (
    TruncatedNormal,
    Constant,
    Normal,
    KaimingUniform,
    XavierUniform,
)

normal_ = Normal(mean=0.0, std=1e-6)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)
kaiming_normal_ = KaimingUniform(nonlinearity="relu")
trunc_normal_ = TruncatedNormal(std=0.02)
xavier_uniform_ = XavierUniform()


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class Conv2dAlign(nn.Conv2D):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(
        self,
        in_channel,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        eps=1e-6,
    ):

        super().__init__(
            in_channel,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias,
            weight_attr=True,
        )
        self.eps = eps

    def forward(self, x):
        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self._stride,
            self._padding,
            self._dilation,
            self._groups,
        )
        return x


class HybridEmbed(nn.Layer):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(
        self,
        backbone,
        img_size=224,
        patch_size=1,
        feature_size=None,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Layer)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        feature_dim = 1024
        feature_size = (42, 12)
        patch_size = (1, 1)
        assert (
            feature_size[0] % patch_size[0] == 0
            and feature_size[1] % patch_size[1] == 0
        )

        self.grid_size = (
            feature_size[0] // patch_size[0],
            feature_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2D(
            feature_dim,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            weight_attr=True,
            bias_attr=True,
        )

    def forward(self, x):

        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose([0, 2, 1])

        return x


class myLinear(nn.Linear):
    def __init__(self, in_channel, out_channels, weight_attr=True, bias_attr=True):
        super().__init__(
            in_channel, out_channels, weight_attr=weight_attr, bias_attr=bias_attr
        )

    def forward(self, x):
        return paddle.matmul(x, self.weight, transpose_y=True) + self.bias


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = myLinear(dim, dim, weight_attr=True, bias_attr=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape([B, N, 3, self.num_heads, C // self.num_heads])
            .transpose([2, 0, 3, 1, 4])
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose([0, 1, 3, 2])) * self.scale

        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Layer):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

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
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Layer):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HybridTransformer(nn.Layer):
    """Implementation of HybridTransformer.

    Args:
      x: input images with shape [N, 1, H, W]
      label: LaTeX-OCR labels with shape [N, L] , L is the max sequence length
      attention_mask: LaTeX-OCR attention mask with shape [N, L]  , L is the max sequence length

    Returns:
      The encoded features with shape [N, 1, H//16, W//16]
    """

    def __init__(
        self,
        backbone_layers=[2, 3, 7],
        input_channel=1,
        is_predict=False,
        is_export=False,
        img_size=(224, 224),
        patch_size=16,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=None,
        norm_layer=None,
        act_layer=None,
        weight_init="",
        **kwargs,
    ):
        super(HybridTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        act_layer = act_layer or nn.GELU
        self.height, self.width = img_size
        self.patch_size = patch_size
        backbone = ResNetV2(
            layers=backbone_layers,
            num_classes=0,
            global_pool="",
            in_chans=input_channel,
            preact=False,
            stem_type="same",
            conv_layer=StdConv2dSame,
            is_export=is_export,
        )
        min_patch_size = 2 ** (len(backbone_layers) + 1)
        self.patch_embed = HybridEmbed(
            img_size=img_size,
            patch_size=patch_size // min_patch_size,
            in_chans=input_channel,
            embed_dim=embed_dim,
            backbone=backbone,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = paddle.create_parameter([1, 1, embed_dim], dtype="float32")
        self.dist_token = (
            paddle.create_parameter(
                [1, 1, embed_dim],
                dtype="float32",
            )
            if distilled
            else None
        )
        self.pos_embed = paddle.create_parameter(
            [1, num_patches + self.num_tokens, embed_dim], dtype="float32"
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        zeros_(self.cls_token)
        if self.dist_token is not None:
            zeros_(self.dist_token)
        zeros_(self.pos_embed)

        dpr = [
            x.item() for x in paddle.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                ("fc", nn.Linear(embed_dim, representation_size)), ("act", nn.Tanh())
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.head_dist = None
        if distilled:
            self.head_dist = (
                nn.Linear(self.embed_dim, self.num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        self.init_weights(weight_init)
        self.out_channels = embed_dim
        self.is_predict = is_predict
        self.is_export = is_export

    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "nlhb", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    def load_pretrained(self, checkpoint_path, prefix=""):
        raise NotImplementedError

    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        if self.num_tokens == 2:
            self.head_dist = (
                nn.Linear(self.embed_dim, self.num_classes)
                if num_classes > 0
                else nn.Identity()
            )

    def forward_features(self, x):
        B, c, h, w = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(
            [B, -1, -1]
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = paddle.concat((cls_tokens, x), axis=1)
        h, w = h // self.patch_size, w // self.patch_size
        repeat_tensor = (
            paddle.arange(h) * (self.width // self.patch_size - w)
        ).reshape([-1, 1])
        repeat_tensor = paddle.repeat_interleave(
            repeat_tensor, paddle.to_tensor(w), axis=1
        ).reshape([-1])
        pos_emb_ind = repeat_tensor + paddle.arange(h * w)
        pos_emb_ind = paddle.concat(
            (paddle.zeros([1], dtype="int64"), pos_emb_ind + 1), axis=0
        ).cast(paddle.int64)
        x += self.pos_embed[:, pos_emb_ind]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, input_data):

        if self.training:
            x, label, attention_mask = input_data
        else:
            if isinstance(input_data, list):
                x = input_data[0]
            else:
                x = input_data
        x = self.forward_features(x)
        x = self.head(x)
        if self.training:
            return x, label, attention_mask
        else:
            return x


def _init_vit_weights(
    module: nn.Layer, name: str = "", head_bias: float = 0.0, jax_impl: bool = False
):
    """ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            zeros_(module.weight)
            constant_ = Constant(value=head_bias)
            constant_(module.bias, head_bias)
        elif name.startswith("pre_logits"):
            zeros_(module.bias)
        else:
            if jax_impl:
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    if "mlp" in name:
                        normal_(module.bias)
                    else:
                        zeros_(module.bias)
            else:
                trunc_normal_(module.weight)
                if module.bias is not None:
                    zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2D):
        # NOTE conv was left to pytorch default in my original init
        if module.bias is not None:
            zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2D)):
        zeros_(module.bias)
        ones_(module.weight)
