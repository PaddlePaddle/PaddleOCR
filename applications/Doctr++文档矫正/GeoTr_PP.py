# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import copy
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from extractor_PP import BasicEncoder
from position_encoding_PP import build_position_encoding
from param_init import multihead_fill, th_linear_fill


class attnLayer(nn.Layer):
    def __init__(
            self,
            d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False, ):
        super().__init__()

        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_list = nn.LayerList([
            copy.deepcopy(
                nn.MultiHeadAttention(
                    d_model, nhead, dropout=dropout)) for i in range(2)
        ])

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2_list = nn.LayerList(
            [copy.deepcopy(nn.LayerNorm(d_model)) for i in range(2)])

        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2_list = nn.LayerList(
            [copy.deepcopy(nn.Dropout(p=dropout)) for i in range(2)])
        self.dropout3 = nn.Dropout(p=dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.init_weight()

    def with_pos_embed(self, tensor, pos: Optional[paddle.Tensor]):
        return tensor if pos is None else tensor + pos

    def init_weight(self):
        multihead_fill(self.self_attn, True)
        for m in self.multihead_attn_list:
            multihead_fill(m, True)
        th_linear_fill(self.linear1)
        th_linear_fill(self.linear2)

    def forward_post(
            self,
            tgt,
            memory_list,
            tgt_mask=None,
            memory_mask=None,
            pos=None,
            memory_pos=None, ):
        q = k = self.with_pos_embed(tgt, pos)
        tgt2 = self.self_attn(
            q.transpose((1, 0, 2)),
            k.transpose((1, 0, 2)),
            value=tgt.transpose((1, 0, 2)),
            attn_mask=tgt_mask)
        tgt2 = tgt2.transpose((1, 0, 2))
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        for memory, multihead_attn, norm2, dropout2, m_pos in zip(
                memory_list,
                self.multihead_attn_list,
                self.norm2_list,
                self.dropout2_list,
                memory_pos, ):
            tgt2 = multihead_attn(
                query=self.with_pos_embed(tgt, pos).transpose((1, 0, 2)),
                key=self.with_pos_embed(memory, m_pos).transpose((1, 0, 2)),
                value=memory.transpose((1, 0, 2)),
                attn_mask=memory_mask, ).transpose((1, 0, 2))

            tgt = tgt + dropout2(tgt2)
            tgt = norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward_pre(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            pos=None,
            memory_pos=None, ):
        tgt2 = self.norm1(tgt)

        q = k = self.with_pos_embed(tgt2, pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(memory, memory_pos),
            value=memory,
            attn_mask=memory_mask, )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(
            self,
            tgt,
            memory_list,
            tgt_mask=None,
            memory_mask=None,
            pos=None,
            memory_pos=None, ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory_list,
                tgt_mask,
                memory_mask,
                pos,
                memory_pos, )
        return self.forward_post(
            tgt,
            memory_list,
            tgt_mask,
            memory_mask,
            pos,
            memory_pos, )


def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransDecoder(nn.Layer):
    def __init__(self, num_attn_layers: int, hidden_dim: int=128):
        super(TransDecoder, self).__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, image: paddle.Tensor, query_embed: paddle.Tensor):
        pos = self.position_embedding(
            paddle.ones(
                [image.shape[0], image.shape[2], image.shape[3]], dtype="bool")
            .cuda())
        b, c, h, w = image.shape

        image = image.flatten(2).transpose(perm=[2, 0, 1])
        pos = pos.flatten(2).transpose(perm=[2, 0, 1])

        for layer in self.layers:
            query_embed = layer(
                query_embed, [image], pos=pos, memory_pos=[pos, pos])

        query_embed = query_embed.transpose(perm=[1, 2, 0]).reshape(
            [b, c, h, w])

        return query_embed


class TransEncoder(nn.Layer):
    def __init__(self, num_attn_layers: int, hidden_dim: int=128):
        super(TransEncoder, self).__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, image: paddle.Tensor):
        pos = self.position_embedding(
            paddle.ones(
                [image.shape[0], image.shape[2], image.shape[3]], dtype="bool")
            .cuda())
        b, c, h, w = image.shape

        image = image.flatten(2).transpose(perm=[2, 0, 1])
        pos = pos.flatten(2).transpose(perm=[2, 0, 1])

        for layer in self.layers:
            image = layer(image, [image], pos=pos, memory_pos=[pos, pos])

        image = image.transpose(perm=[1, 2, 0]).reshape([b, c, h, w])

        return image


class FlowHead(nn.Layer):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()

        self.conv1 = nn.Conv2D(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2D(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UpdateBlock(nn.Layer):
    def __init__(self, hidden_dim: int=128):
        super(UpdateBlock, self).__init__()

        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2D(
                hidden_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2D(
                256, 64 * 9, 1, padding=0), )

    def forward(self, image, coords):
        mask = 0.25 * self.mask(image)
        dflow = self.flow_head(image)
        coords = coords + dflow
        return mask, coords


def coords_grid(batch, ht, wd):
    coords = paddle.meshgrid(paddle.arange(end=ht), paddle.arange(end=wd))
    coords = paddle.stack(coords[::-1], axis=0).astype(dtype="float32")
    return coords[None].tile([batch, 1, 1, 1])


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class OverlapPatchEmbed(nn.Layer):
    """Image to Patch Embedding"""

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()

        img_size = img_size if isinstance(img_size, tuple) else (img_size,
                                                                 img_size)
        patch_size = patch_size if isinstance(patch_size, tuple) else (
            patch_size, patch_size)

        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2), )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            weight_init_(m, "trunc_normal_", std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2D):
            weight_init_(
                m.weight,
                "kaiming_normal_",
                mode="fan_out",
                nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2)

        perm = list(range(x.ndim))
        perm[1] = 2
        perm[2] = 1
        x = x.transpose(perm=perm)

        x = self.norm(x)

        return x, H, W


class GeoTr(nn.Layer):
    def __init__(self):
        super(GeoTr, self).__init__()

        self.hidden_dim = hdim = 256

        self.fnet = BasicEncoder(output_dim=hdim, norm_fn="instance")

        self.encoder_block = [("encoder_block" + str(i)) for i in range(3)]
        for i in self.encoder_block:
            self.__setattr__(i, TransEncoder(2, hidden_dim=hdim))

        self.down_layer = [("down_layer" + str(i)) for i in range(2)]
        for i in self.down_layer:
            self.__setattr__(i, nn.Conv2D(256, 256, 3, stride=2, padding=1))

        self.decoder_block = [("decoder_block" + str(i)) for i in range(3)]
        for i in self.decoder_block:
            self.__setattr__(i, TransDecoder(2, hidden_dim=hdim))

        self.up_layer = [("up_layer" + str(i)) for i in range(2)]
        for i in self.up_layer:
            self.__setattr__(
                i,
                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True))

        self.query_embed = nn.Embedding(81, self.hidden_dim)

        self.update_block = UpdateBlock(self.hidden_dim)

    def initialize_flow(self, img):
        N, _, H, W = img.shape
        coodslar = coords_grid(N, H, W)
        coords0 = coords_grid(N, H // 8, W // 8)
        coords1 = coords_grid(N, H // 8, W // 8)
        return coodslar, coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape

        mask = mask.reshape([N, 1, 9, 8, 8, H, W])
        mask = F.softmax(mask, axis=2)

        up_flow = F.unfold(8 * flow, [3, 3], paddings=1)
        up_flow = up_flow.reshape([N, 2, 9, 1, 1, H, W])

        up_flow = paddle.sum(mask * up_flow, axis=2)
        up_flow = up_flow.transpose(perm=[0, 1, 4, 2, 5, 3])

        return up_flow.reshape([N, 2, 8 * H, 8 * W])

    def forward(self, image):
        fmap = self.fnet(image)
        fmap = F.relu(fmap)

        fmap1 = self.__getattr__(self.encoder_block[0])(fmap)
        fmap1d = self.__getattr__(self.down_layer[0])(fmap1)
        fmap2 = self.__getattr__(self.encoder_block[1])(fmap1d)
        fmap2d = self.__getattr__(self.down_layer[1])(fmap2)
        fmap3 = self.__getattr__(self.encoder_block[2])(fmap2d)

        query_embed0 = self.query_embed.weight.unsqueeze(1).tile(
            [1, fmap3.shape[0], 1])

        fmap3d_ = self.__getattr__(self.decoder_block[0])(fmap3, query_embed0)
        fmap3du_ = (self.__getattr__(self.up_layer[0])(fmap3d_).flatten(2)
                    .transpose(perm=[2, 0, 1]))
        fmap2d_ = self.__getattr__(self.decoder_block[1])(fmap2, fmap3du_)
        fmap2du_ = (self.__getattr__(self.up_layer[1])(fmap2d_).flatten(2)
                    .transpose(perm=[2, 0, 1]))
        fmap_out = self.__getattr__(self.decoder_block[2])(fmap1, fmap2du_)

        coodslar, coords0, coords1 = self.initialize_flow(image)
        coords1 = coords1.detach()
        mask, coords1 = self.update_block(fmap_out, coords1)
        flow_up = self.upsample_flow(coords1 - coords0, mask)
        bm_up = coodslar + flow_up

        return bm_up
