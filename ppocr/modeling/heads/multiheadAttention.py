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

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.nn import Linear
from paddle.nn.initializer import XavierUniform as xavier_uniform_
from paddle.nn.initializer import Constant as constant_
from paddle.nn.initializer import XavierNormal as xavier_normal_

zeros_ = constant_(value=0.)
ones_ = constant_(value=1.)


class MultiheadAttention(nn.Layer):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model
        num_heads: parallel attention layers, or heads

    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.out_proj = Linear(embed_dim, embed_dim, bias_attr=bias)
        self._reset_parameters()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1))
        self.conv2 = paddle.nn.Conv2D(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1))
        self.conv3 = paddle.nn.Conv2D(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1))

    def _reset_parameters(self):
        xavier_uniform_(self.out_proj.weight)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                incremental_state=None,
                attn_mask=None):
        """
        Inputs of forward function
            query: [target length, batch size, embed dim]
            key: [sequence length, batch size, embed dim]
            value: [sequence length, batch size, embed dim]
            key_padding_mask: if True, mask padding based on batch size
            incremental_state: if provided, previous time steps are cashed
            need_weights: output attn_output_weights
            static_kv: key and value are static

        Outputs of forward function
            attn_output: [target length, batch size, embed dim]
            attn_output_weights: [batch size, target length, sequence length]
        """
        q_shape = paddle.shape(query)
        src_shape = paddle.shape(key)
        q = self._in_proj_q(query)
        k = self._in_proj_k(key)
        v = self._in_proj_v(value)
        q *= self.scaling
        q = paddle.transpose(
            paddle.reshape(
                q, [q_shape[0], q_shape[1], self.num_heads, self.head_dim]),
            [1, 2, 0, 3])
        k = paddle.transpose(
            paddle.reshape(
                k, [src_shape[0], q_shape[1], self.num_heads, self.head_dim]),
            [1, 2, 0, 3])
        v = paddle.transpose(
            paddle.reshape(
                v, [src_shape[0], q_shape[1], self.num_heads, self.head_dim]),
            [1, 2, 0, 3])
        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == q_shape[1]
            assert key_padding_mask.shape[1] == src_shape[0]
        attn_output_weights = paddle.matmul(q,
                                            paddle.transpose(k, [0, 1, 3, 2]))
        if attn_mask is not None:
            attn_mask = paddle.unsqueeze(paddle.unsqueeze(attn_mask, 0), 0)
            attn_output_weights += attn_mask
        if key_padding_mask is not None:
            attn_output_weights = paddle.reshape(
                attn_output_weights,
                [q_shape[1], self.num_heads, q_shape[0], src_shape[0]])
            key = paddle.unsqueeze(paddle.unsqueeze(key_padding_mask, 1), 2)
            key = paddle.cast(key, 'float32')
            y = paddle.full(
                shape=paddle.shape(key), dtype='float32', fill_value='-inf')
            y = paddle.where(key == 0., key, y)
            attn_output_weights += y
        attn_output_weights = F.softmax(
            attn_output_weights.astype('float32'),
            axis=-1,
            dtype=paddle.float32 if attn_output_weights.dtype == paddle.float16
            else attn_output_weights.dtype)
        attn_output_weights = F.dropout(
            attn_output_weights, p=self.dropout, training=self.training)

        attn_output = paddle.matmul(attn_output_weights, v)
        attn_output = paddle.reshape(
            paddle.transpose(attn_output, [2, 0, 1, 3]),
            [q_shape[0], q_shape[1], self.embed_dim])
        attn_output = self.out_proj(attn_output)

        return attn_output

    def _in_proj_q(self, query):
        query = paddle.transpose(query, [1, 2, 0])
        query = paddle.unsqueeze(query, axis=2)
        res = self.conv1(query)
        res = paddle.squeeze(res, axis=2)
        res = paddle.transpose(res, [2, 0, 1])
        return res

    def _in_proj_k(self, key):
        key = paddle.transpose(key, [1, 2, 0])
        key = paddle.unsqueeze(key, axis=2)
        res = self.conv2(key)
        res = paddle.squeeze(res, axis=2)
        res = paddle.transpose(res, [2, 0, 1])
        return res

    def _in_proj_v(self, value):
        value = paddle.transpose(value, [1, 2, 0])  #(1, 2, 0)
        value = paddle.unsqueeze(value, axis=2)
        res = self.conv3(value)
        res = paddle.squeeze(res, axis=2)
        res = paddle.transpose(res, [2, 0, 1])
        return res
