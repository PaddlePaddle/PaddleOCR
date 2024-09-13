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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn
from paddle import ParamAttr
import paddle.nn.functional as F
import numpy as np

from .rec_att_head import AttentionGRUCell
from ppocr.modeling.backbones.rec_svtrnet import DropPath, Identity, Mlp


def get_para_bias_attr(l2_decay, k):
    if l2_decay > 0:
        regularizer = paddle.regularizer.L2Decay(l2_decay)
        stdv = 1.0 / math.sqrt(k * 1.0)
        initializer = nn.initializer.Uniform(-stdv, stdv)
    else:
        regularizer = None
        initializer = None
    weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    return [weight_attr, bias_attr]


class TableAttentionHead(nn.Layer):
    def __init__(
        self,
        in_channels,
        hidden_size,
        in_max_len=488,
        max_text_length=800,
        out_channels=30,
        loc_reg_num=4,
        **kwargs,
    ):
        super(TableAttentionHead, self).__init__()
        self.input_size = in_channels[-1]
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.max_text_length = max_text_length

        self.structure_attention_cell = AttentionGRUCell(
            self.input_size, hidden_size, self.out_channels, use_gru=False
        )
        self.structure_generator = nn.Linear(hidden_size, self.out_channels)
        self.in_max_len = in_max_len

        if self.in_max_len == 640:
            self.loc_fea_trans = nn.Linear(400, self.max_text_length + 1)
        elif self.in_max_len == 800:
            self.loc_fea_trans = nn.Linear(625, self.max_text_length + 1)
        else:
            self.loc_fea_trans = nn.Linear(256, self.max_text_length + 1)
        self.loc_generator = nn.Linear(self.input_size + hidden_size, loc_reg_num)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None):
        # if and else branch are both needed when you want to assign a variable
        # if you modify the var in just one branch, then the modification will not work.
        fea = inputs[-1]
        last_shape = int(np.prod(fea.shape[2:]))  # gry added
        fea = paddle.reshape(fea, [fea.shape[0], fea.shape[1], last_shape])
        fea = fea.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
        batch_size = fea.shape[0]

        hidden = paddle.zeros((batch_size, self.hidden_size))
        output_hiddens = paddle.zeros(
            (batch_size, self.max_text_length + 1, self.hidden_size)
        )
        if self.training and targets is not None:
            structure = targets[0]
            for i in range(self.max_text_length + 1):
                elem_onehots = self._char_to_onehot(
                    structure[:, i], onehot_dim=self.out_channels
                )
                (outputs, hidden), alpha = self.structure_attention_cell(
                    hidden, fea, elem_onehots
                )
                output_hiddens[:, i, :] = outputs
            structure_probs = self.structure_generator(output_hiddens)
            loc_fea = fea.transpose([0, 2, 1])
            loc_fea = self.loc_fea_trans(loc_fea)
            loc_fea = loc_fea.transpose([0, 2, 1])
            loc_concat = paddle.concat([output_hiddens, loc_fea], axis=2)
            loc_preds = self.loc_generator(loc_concat)
            loc_preds = F.sigmoid(loc_preds)
        else:
            temp_elem = paddle.zeros(shape=[batch_size], dtype="int32")
            structure_probs = None
            loc_preds = None
            elem_onehots = None
            outputs = None
            alpha = None
            max_text_length = paddle.to_tensor(self.max_text_length)
            for i in range(max_text_length + 1):
                elem_onehots = self._char_to_onehot(
                    temp_elem, onehot_dim=self.out_channels
                )
                (outputs, hidden), alpha = self.structure_attention_cell(
                    hidden, fea, elem_onehots
                )
                output_hiddens[:, i, :] = outputs
                structure_probs_step = self.structure_generator(outputs)
                temp_elem = structure_probs_step.argmax(axis=1, dtype="int32")

            structure_probs = self.structure_generator(output_hiddens)
            structure_probs = F.softmax(structure_probs)
            loc_fea = fea.transpose([0, 2, 1])
            loc_fea = self.loc_fea_trans(loc_fea)
            loc_fea = loc_fea.transpose([0, 2, 1])
            loc_concat = paddle.concat([output_hiddens, loc_fea], axis=2)
            loc_preds = self.loc_generator(loc_concat)
            loc_preds = F.sigmoid(loc_preds)
        return {"structure_probs": structure_probs, "loc_preds": loc_preds}


class HWAttention(nn.Layer):
    def __init__(
        self,
        head_dim=32,
        qk_scale=None,
        attn_drop=0.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or self.head_dim**-0.5
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        C = C // 3
        qkv = x.reshape([B, N, 3, C // self.head_dim, self.head_dim]).transpose(
            [2, 0, 3, 1, 4]
        )
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose([0, 1, 3, 2]) * self.scale
        attn = F.softmax(attn, -1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose([0, 2, 1]).reshape([B, N, C])
        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, H, W, C = img.shape
    img_reshape = img.reshape([B, H // H_sp, H_sp, W // W_sp, W_sp, C])
    img_perm = img_reshape.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, H_sp * W_sp, C])
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.reshape([B, H // H_sp, W // W_sp, H_sp, W_sp, -1])
    img = img.transpose([0, 1, 3, 2, 4, 5]).flatten(1, 4)
    return img


class Block(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        split_h=4,
        split_w=4,
        h_num_heads=None,
        w_num_heads=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eps=1e-6,
    ):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.split_h = split_h
        self.split_w = split_w
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, epsilon=eps)
        self.h_num_heads = h_num_heads if h_num_heads is not None else num_heads // 2
        self.w_num_heads = w_num_heads if w_num_heads is not None else num_heads // 2
        self.head_dim = dim // num_heads
        self.mixer = HWAttention(
            head_dim=dim // num_heads,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, epsilon=eps)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose([0, 2, 1])

        qkv = self.qkv(x).reshape([B, H, W, 3 * C])

        x1 = qkv[:, :, :, : 3 * self.h_num_heads * self.head_dim]  # b, h, w, 3ch
        x2 = qkv[:, :, :, 3 * self.h_num_heads * self.head_dim :]  # b, h, w, 3cw

        x1 = self.mixer(img2windows(x1, self.split_h, W))  # b*splith, W, 3ch
        x2 = self.mixer(img2windows(x2, H, self.split_w))  # b*splitw, h, 3ch
        x1 = windows2img(x1, self.split_h, W, H, W)
        x2 = windows2img(x2, H, self.split_w, H, W)

        attened_x = paddle.concat([x1, x2], 2)
        attened_x = self.proj(attened_x)

        x = self.norm1(x + self.drop_path(attened_x))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        x = x.transpose([0, 2, 1]).reshape([-1, C, H, W])
        return x


class SLAHead(nn.Layer):
    def __init__(
        self,
        in_channels,
        hidden_size,
        out_channels=30,
        max_text_length=500,
        loc_reg_num=4,
        fc_decay=0.0,
        use_attn=False,
        **kwargs,
    ):
        """
        @param in_channels: input shape
        @param hidden_size: hidden_size for RNN and Embedding
        @param out_channels: num_classes to rec
        @param max_text_length: max text pred
        """
        super().__init__()
        in_channels = in_channels[-1]
        self.hidden_size = hidden_size
        self.max_text_length = max_text_length
        self.emb = self._char_to_onehot
        self.num_embeddings = out_channels
        self.loc_reg_num = loc_reg_num
        self.eos = self.num_embeddings - 1

        # structure
        self.structure_attention_cell = AttentionGRUCell(
            in_channels, hidden_size, self.num_embeddings
        )
        weight_attr, bias_attr = get_para_bias_attr(l2_decay=fc_decay, k=hidden_size)
        weight_attr1_1, bias_attr1_1 = get_para_bias_attr(
            l2_decay=fc_decay, k=hidden_size
        )
        weight_attr1_2, bias_attr1_2 = get_para_bias_attr(
            l2_decay=fc_decay, k=hidden_size
        )
        self.structure_generator = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                self.hidden_size,
                weight_attr=weight_attr1_2,
                bias_attr=bias_attr1_2,
            ),
            nn.Linear(
                hidden_size, out_channels, weight_attr=weight_attr, bias_attr=bias_attr
            ),
        )
        dpr = np.linspace(0, 0.1, 2)

        self.use_attn = use_attn
        if use_attn:
            layer_list = [
                Block(
                    in_channels,
                    num_heads=2,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop_path=dpr[i],
                )
                for i in range(2)
            ]
            self.cross_atten = nn.Sequential(*layer_list)
        # loc
        weight_attr1, bias_attr1 = get_para_bias_attr(
            l2_decay=fc_decay, k=self.hidden_size
        )
        weight_attr2, bias_attr2 = get_para_bias_attr(
            l2_decay=fc_decay, k=self.hidden_size
        )
        self.loc_generator = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                self.hidden_size,
                weight_attr=weight_attr1,
                bias_attr=bias_attr1,
            ),
            nn.Linear(
                self.hidden_size,
                loc_reg_num,
                weight_attr=weight_attr2,
                bias_attr=bias_attr2,
            ),
            nn.Sigmoid(),
        )

    def forward(self, inputs, targets=None):
        fea = inputs[-1]
        batch_size = fea.shape[0]
        if self.use_attn:
            fea = fea + self.cross_atten(fea)
        # reshape
        fea = paddle.reshape(fea, [fea.shape[0], fea.shape[1], -1])
        fea = fea.transpose([0, 2, 1])  # (NTC)(batch, width, channels)

        hidden = paddle.zeros((batch_size, self.hidden_size))
        structure_preds = paddle.zeros(
            (batch_size, self.max_text_length + 1, self.num_embeddings)
        )
        loc_preds = paddle.zeros(
            (batch_size, self.max_text_length + 1, self.loc_reg_num)
        )
        structure_preds.stop_gradient = True
        loc_preds.stop_gradient = True

        if self.training and targets is not None:
            structure = targets[0]
            max_len = targets[-2].max().astype("int32")
            for i in range(max_len + 1):
                hidden, structure_step, loc_step = self._decode(
                    structure[:, i], fea, hidden
                )
                structure_preds[:, i, :] = structure_step
                loc_preds[:, i, :] = loc_step
            structure_preds = structure_preds[:, : max_len + 1]
            loc_preds = loc_preds[:, : max_len + 1]
        else:
            structure_ids = paddle.zeros(
                (batch_size, self.max_text_length + 1), dtype="int32"
            )
            pre_chars = paddle.zeros(shape=[batch_size], dtype="int32")
            max_text_length = paddle.to_tensor(self.max_text_length)
            for i in range(max_text_length + 1):
                hidden, structure_step, loc_step = self._decode(pre_chars, fea, hidden)
                pre_chars = structure_step.argmax(axis=1, dtype="int32")
                structure_preds[:, i, :] = structure_step
                loc_preds[:, i, :] = loc_step

                structure_ids[:, i] = pre_chars
                if (structure_ids == self.eos).any(-1).all():
                    break
        if not self.training:
            structure_preds = F.softmax(structure_preds[:, : i + 1])
            loc_preds = loc_preds[:, : i + 1]
        return {"structure_probs": structure_preds, "loc_preds": loc_preds}

    def _decode(self, pre_chars, features, hidden):
        """
        Predict table label and coordinates for each step
        @param pre_chars: Table label in previous step
        @param features:
        @param hidden: hidden status in previous step
        @return:
        """
        emb_feature = self.emb(pre_chars)
        # output shape is b * self.hidden_size
        (output, hidden), alpha = self.structure_attention_cell(
            hidden, features, emb_feature
        )

        # structure
        structure_step = self.structure_generator(output)
        # loc
        loc_step = self.loc_generator(output)
        return hidden, structure_step, loc_step

    def _char_to_onehot(self, input_char):
        input_ont_hot = F.one_hot(input_char, self.num_embeddings)
        return input_ont_hot
