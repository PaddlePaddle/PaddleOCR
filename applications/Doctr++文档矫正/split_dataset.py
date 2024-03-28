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

import math
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.initializer as init


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[paddle.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class PositionEmbeddingSine(nn.Layer):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi

        self.scale = scale

    def forward(self, mask):
        assert mask is not None

        y_embed = mask.cumsum(axis=1, dtype="float32")
        x_embed = mask.cumsum(axis=2, dtype="float32")

        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = paddle.arange(end=self.num_pos_feats, dtype="float32")
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = paddle.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            axis=4).flatten(3)
        pos_y = paddle.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            axis=4).flatten(3)

        pos = paddle.concat((pos_y, pos_x), axis=3).transpose(perm=[0, 3, 1, 2])

        return pos


class PositionEmbeddingLearned(nn.Layer):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        init_Constant = init.Uniform()
        init_Constant(self.row_embed.weight)
        init_Constant(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors

        h, w = x.shape[-2:]

        i = paddle.arange(end=w)
        j = paddle.arange(end=h)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = (paddle.concat(
            [
                x_emb.unsqueeze(0).tile([h, 1, 1]),
                y_emb.unsqueeze(1).tile([1, w, 1]),
            ],
            axis=-1, ).transpose([2, 0, 1]).unsqueeze(0)
               .tile([x.shape[0], 1, 1, 1]))

        return pos


def build_position_encoding(hidden_dim=512, position_embedding="sine"):
    N_steps = hidden_dim // 2

    if position_embedding in ("v2", "sine"):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {position_embedding}")
    return position_embedding
