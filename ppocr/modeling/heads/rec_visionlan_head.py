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
"""
This code is refer from:
https://github.com/wangyuxin87/VisionLAN
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, XavierNormal
import numpy as np


class PositionalEncoding(nn.Layer):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = paddle.to_tensor(sinusoid_table, dtype="float32")
        sinusoid_table = paddle.unsqueeze(sinusoid_table, axis=0)
        return sinusoid_table

    def forward(self, x):
        return x + self.pos_table[:, : x.shape[1]].clone().detach()


class ScaledDotProductAttention(nn.Layer):
    "Scaled Dot-Product Attention"

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(axis=2)

    def forward(self, q, k, v, mask=None):
        k = paddle.transpose(k, perm=[0, 2, 1])
        attn = paddle.bmm(q, k)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
            if mask.dim() == 3:
                mask = paddle.unsqueeze(mask, axis=1)
            elif mask.dim() == 2:
                mask = paddle.unsqueeze(mask, axis=1)
                mask = paddle.unsqueeze(mask, axis=1)
            repeat_times = [
                attn.shape[1] // mask.shape[1],
                attn.shape[2] // mask.shape[2],
            ]
            mask = paddle.tile(mask, [1, repeat_times[0], repeat_times[1], 1])
            attn[mask == 0] = -1e9
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = paddle.bmm(attn, v)
        return output


class MultiHeadAttention(nn.Layer):
    "Multi-Head Attention module"

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(
            d_model,
            n_head * d_k,
            weight_attr=ParamAttr(
                initializer=Normal(mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
            ),
        )
        self.w_ks = nn.Linear(
            d_model,
            n_head * d_k,
            weight_attr=ParamAttr(
                initializer=Normal(mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
            ),
        )
        self.w_vs = nn.Linear(
            d_model,
            n_head * d_v,
            weight_attr=ParamAttr(
                initializer=Normal(mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
            ),
        )

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(
            n_head * d_v, d_model, weight_attr=ParamAttr(initializer=XavierNormal())
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape
        residual = q

        q = self.w_qs(q)
        q = paddle.reshape(q, shape=[-1, len_q, n_head, d_k])  # 4*21*512 ---- 4*21*8*64
        k = self.w_ks(k)
        k = paddle.reshape(k, shape=[-1, len_k, n_head, d_k])
        v = self.w_vs(v)
        v = paddle.reshape(v, shape=[-1, len_v, n_head, d_v])

        q = paddle.transpose(q, perm=[2, 0, 1, 3])
        q = paddle.reshape(q, shape=[-1, len_q, d_k])  # (n*b) x lq x dk
        k = paddle.transpose(k, perm=[2, 0, 1, 3])
        k = paddle.reshape(k, shape=[-1, len_k, d_k])  # (n*b) x lk x dk
        v = paddle.transpose(v, perm=[2, 0, 1, 3])
        v = paddle.reshape(v, shape=[-1, len_v, d_v])  # (n*b) x lv x dv

        mask = (
            paddle.tile(mask, [n_head, 1, 1]) if mask is not None else None
        )  # (n*b) x .. x ..
        output = self.attention(q, k, v, mask=mask)
        output = paddle.reshape(output, shape=[n_head, -1, len_q, d_v])
        output = paddle.transpose(output, perm=[1, 2, 0, 3])
        output = paddle.reshape(
            output, shape=[-1, len_q, n_head * d_v]
        )  # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output


class PositionwiseFeedForward(nn.Layer):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1D(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1D(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = paddle.transpose(x, perm=[0, 2, 1])
        x = self.w_2(F.relu(self.w_1(x)))
        x = paddle.transpose(x, perm=[0, 2, 1])
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


class EncoderLayer(nn.Layer):
    """Compose with two layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class Transformer_Encoder(nn.Layer):
    def __init__(
        self,
        n_layers=2,
        n_head=8,
        d_word_vec=512,
        d_k=64,
        d_v=64,
        d_model=512,
        d_inner=2048,
        dropout=0.1,
        n_position=256,
    ):
        super(Transformer_Encoder, self).__init__()
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.LayerList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model, epsilon=1e-6)

    def forward(self, enc_output, src_mask, return_attns=False):
        enc_output = self.dropout(self.position_enc(enc_output))  # position embedding
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, slf_attn_mask=src_mask)
        enc_output = self.layer_norm(enc_output)
        return enc_output


class PP_layer(nn.Layer):
    def __init__(self, n_dim=512, N_max_character=25, n_position=256):
        super(PP_layer, self).__init__()
        self.character_len = N_max_character
        self.f0_embedding = nn.Embedding(N_max_character, n_dim)
        self.w0 = nn.Linear(N_max_character, n_position)
        self.wv = nn.Linear(n_dim, n_dim)
        self.we = nn.Linear(n_dim, N_max_character)
        self.active = nn.Tanh()
        self.softmax = nn.Softmax(axis=2)

    def forward(self, enc_output):
        # enc_output: b,256,512
        reading_order = paddle.arange(self.character_len, dtype="int64")
        reading_order = reading_order.unsqueeze(0).expand(
            [enc_output.shape[0], self.character_len]
        )  # (S,) -> (B, S)
        reading_order = self.f0_embedding(reading_order)  # b,25,512

        # calculate attention
        reading_order = paddle.transpose(reading_order, perm=[0, 2, 1])
        t = self.w0(reading_order)  # b,512,256
        t = self.active(
            paddle.transpose(t, perm=[0, 2, 1]) + self.wv(enc_output)
        )  # b,256,512
        t = self.we(t)  # b,256,25
        t = self.softmax(paddle.transpose(t, perm=[0, 2, 1]))  # b,25,256
        g_output = paddle.bmm(t, enc_output)  # b,25,512
        return g_output


class Prediction(nn.Layer):
    def __init__(self, n_dim=512, n_position=256, N_max_character=25, n_class=37):
        super(Prediction, self).__init__()
        self.pp = PP_layer(
            n_dim=n_dim, N_max_character=N_max_character, n_position=n_position
        )
        self.pp_share = PP_layer(
            n_dim=n_dim, N_max_character=N_max_character, n_position=n_position
        )
        self.w_vrm = nn.Linear(n_dim, n_class)  # output layer
        self.w_share = nn.Linear(n_dim, n_class)  # output layer
        self.nclass = n_class

    def forward(self, cnn_feature, f_res, f_sub, train_mode=False, use_mlm=True):
        if train_mode:
            if not use_mlm:
                g_output = self.pp(cnn_feature)  # b,25,512
                g_output = self.w_vrm(g_output)
                f_res = 0
                f_sub = 0
                return g_output, f_res, f_sub
            g_output = self.pp(cnn_feature)  # b,25,512
            f_res = self.pp_share(f_res)
            f_sub = self.pp_share(f_sub)
            g_output = self.w_vrm(g_output)
            f_res = self.w_share(f_res)
            f_sub = self.w_share(f_sub)
            return g_output, f_res, f_sub
        else:
            g_output = self.pp(cnn_feature)  # b,25,512
            g_output = self.w_vrm(g_output)
            return g_output


class MLM(nn.Layer):
    "Architecture of MLM"

    def __init__(self, n_dim=512, n_position=256, max_text_length=25):
        super(MLM, self).__init__()
        self.MLM_SequenceModeling_mask = Transformer_Encoder(
            n_layers=2, n_position=n_position
        )
        self.MLM_SequenceModeling_WCL = Transformer_Encoder(
            n_layers=1, n_position=n_position
        )
        self.pos_embedding = nn.Embedding(max_text_length, n_dim)
        self.w0_linear = nn.Linear(1, n_position)
        self.wv = nn.Linear(n_dim, n_dim)
        self.active = nn.Tanh()
        self.we = nn.Linear(n_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, label_pos):
        # transformer unit for generating mask_c
        feature_v_seq = self.MLM_SequenceModeling_mask(x, src_mask=None)
        # position embedding layer
        label_pos = paddle.to_tensor(label_pos, dtype="int64")
        pos_emb = self.pos_embedding(label_pos)
        pos_emb = self.w0_linear(paddle.unsqueeze(pos_emb, axis=2))
        pos_emb = paddle.transpose(pos_emb, perm=[0, 2, 1])
        # fusion position embedding with features V & generate mask_c
        att_map_sub = self.active(pos_emb + self.wv(feature_v_seq))
        att_map_sub = self.we(att_map_sub)  # b,256,1
        att_map_sub = paddle.transpose(att_map_sub, perm=[0, 2, 1])
        att_map_sub = self.sigmoid(att_map_sub)  # b,1,256
        # WCL
        ## generate inputs for WCL
        att_map_sub = paddle.transpose(att_map_sub, perm=[0, 2, 1])
        f_res = x * (1 - att_map_sub)  # second path with remaining string
        f_sub = x * att_map_sub  # first path with occluded character
        ## transformer units in WCL
        f_res = self.MLM_SequenceModeling_WCL(f_res, src_mask=None)
        f_sub = self.MLM_SequenceModeling_WCL(f_sub, src_mask=None)
        return f_res, f_sub, att_map_sub


def trans_1d_2d(x):
    b, w_h, c = x.shape  # b, 256, 512
    x = paddle.transpose(x, perm=[0, 2, 1])
    x = paddle.reshape(x, [-1, c, 32, 8])
    x = paddle.transpose(x, perm=[0, 1, 3, 2])  # [b, c, 8, 32]
    return x


class MLM_VRM(nn.Layer):
    """
    MLM+VRM, MLM is only used in training.
    ratio controls the occluded number in a batch.
    The pipeline of VisionLAN in testing is very concise with only a backbone + sequence modeling(transformer unit) + prediction layer(pp layer).
    x: input image
    label_pos: character index
    training_step: LF or LA process
    output
    text_pre: prediction of VRM
    test_rem: prediction of remaining string in MLM
    text_mas: prediction of occluded character in MLM
    mask_c_show: visualization of Mask_c
    """

    def __init__(
        self, n_layers=3, n_position=256, n_dim=512, max_text_length=25, nclass=37
    ):
        super(MLM_VRM, self).__init__()
        self.MLM = MLM(
            n_dim=n_dim, n_position=n_position, max_text_length=max_text_length
        )
        self.SequenceModeling = Transformer_Encoder(
            n_layers=n_layers, n_position=n_position
        )
        self.Prediction = Prediction(
            n_dim=n_dim,
            n_position=n_position,
            N_max_character=max_text_length
            + 1,  # N_max_character = 1 eos + 25 characters
            n_class=nclass,
        )
        self.nclass = nclass
        self.max_text_length = max_text_length

    def forward(self, x, label_pos, training_step, train_mode=False):
        b, c, h, w = x.shape
        nT = self.max_text_length
        x = paddle.transpose(x, perm=[0, 1, 3, 2])
        x = paddle.reshape(x, [-1, c, h * w])
        x = paddle.transpose(x, perm=[0, 2, 1])
        if train_mode:
            if training_step == "LF_1":
                f_res = 0
                f_sub = 0
                x = self.SequenceModeling(x, src_mask=None)
                text_pre, test_rem, text_mas = self.Prediction(
                    x, f_res, f_sub, train_mode=True, use_mlm=False
                )
                return text_pre, text_pre, text_pre, text_pre
            elif training_step == "LF_2":
                # MLM
                f_res, f_sub, mask_c = self.MLM(x, label_pos)
                x = self.SequenceModeling(x, src_mask=None)
                text_pre, test_rem, text_mas = self.Prediction(
                    x, f_res, f_sub, train_mode=True
                )
                mask_c_show = trans_1d_2d(mask_c)
                return text_pre, test_rem, text_mas, mask_c_show
            elif training_step == "LA":
                # MLM
                f_res, f_sub, mask_c = self.MLM(x, label_pos)
                ## use the mask_c (1 for occluded character and 0 for remaining characters) to occlude input
                ## ratio controls the occluded number in a batch
                character_mask = paddle.zeros_like(mask_c)

                ratio = b // 2
                if ratio >= 1:
                    with paddle.no_grad():
                        character_mask[0:ratio, :, :] = mask_c[0:ratio, :, :]
                else:
                    character_mask = mask_c
                x = x * (1 - character_mask)
                # VRM
                ## transformer unit for VRM
                x = self.SequenceModeling(x, src_mask=None)
                ## prediction layer for MLM and VSR
                text_pre, test_rem, text_mas = self.Prediction(
                    x, f_res, f_sub, train_mode=True
                )
                mask_c_show = trans_1d_2d(mask_c)
                return text_pre, test_rem, text_mas, mask_c_show
            else:
                raise NotImplementedError
        else:  # VRM is only used in the testing stage
            f_res = 0
            f_sub = 0
            contextual_feature = self.SequenceModeling(x, src_mask=None)
            text_pre = self.Prediction(
                contextual_feature, f_res, f_sub, train_mode=False, use_mlm=False
            )
            text_pre = paddle.transpose(text_pre, perm=[1, 0, 2])  # (26, b, 37))
            return text_pre, x


class VLHead(nn.Layer):
    """
    Architecture of VisionLAN
    """

    def __init__(
        self,
        in_channels,
        out_channels=36,
        n_layers=3,
        n_position=256,
        n_dim=512,
        max_text_length=25,
        training_step="LA",
    ):
        super(VLHead, self).__init__()
        self.MLM_VRM = MLM_VRM(
            n_layers=n_layers,
            n_position=n_position,
            n_dim=n_dim,
            max_text_length=max_text_length,
            nclass=out_channels + 1,
        )
        self.training_step = training_step

    def forward(self, feat, targets=None):
        if self.training:
            label_pos = targets[-2]
            text_pre, test_rem, text_mas, mask_map = self.MLM_VRM(
                feat, label_pos, self.training_step, train_mode=True
            )
            return text_pre, test_rem, text_mas, mask_map
        else:
            text_pre, x = self.MLM_VRM(
                feat, targets, self.training_step, train_mode=False
            )
            return text_pre, x
