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
https://github.com/open-mmlab/mmocr/blob/1.x/mmocr/models/textrecog/encoders/satrn_encoder.py
https://github.com/open-mmlab/mmocr/blob/1.x/mmocr/models/textrecog/decoders/nrtr_decoder.py
"""

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr, reshape, transpose
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import KaimingNormal, Uniform, Constant


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 num_groups=1):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False)

        self.bn = nn.BatchNorm2D(
            num_filters,
            weight_attr=ParamAttr(initializer=Constant(1)),
            bias_attr=ParamAttr(initializer=Constant(0)))
        self.relu = nn.ReLU()

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.bn(y)
        y = self.relu(y)
        return y


class SATRNEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model=512,
                 d_inner=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, qkv_bias=qkv_bias, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = LocalityAwareFeedforward(
            d_model, d_inner, dropout=dropout)

    def forward(self, x, h, w, mask=None):
        n, hw, c = x.shape
        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x, x, x, mask)
        residual = x
        x = self.norm2(x)
        x = x.transpose([0, 2, 1]).reshape([n, c, h, w])
        x = self.feed_forward(x)
        x = x.reshape([n, c, hw]).transpose([0, 2, 1])
        x = residual + x
        return x


class LocalityAwareFeedforward(nn.Layer):
    def __init__(
            self,
            d_in,
            d_hid,
            dropout=0.1, ):
        super().__init__()
        self.conv1 = ConvBNLayer(d_in, 1, d_hid, stride=1, padding=0)

        self.depthwise_conv = ConvBNLayer(
            d_hid, 3, d_hid, stride=1, padding=1, num_groups=d_hid)

        self.conv2 = ConvBNLayer(d_hid, 1, d_in, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.conv2(x)

        return x


class Adaptive2DPositionalEncoding(nn.Layer):
    def __init__(self, d_hid=512, n_height=100, n_width=100, dropout=0.1):
        super().__init__()

        h_position_encoder = self._get_sinusoid_encoding_table(n_height, d_hid)
        h_position_encoder = h_position_encoder.transpose([1, 0])
        h_position_encoder = h_position_encoder.reshape([1, d_hid, n_height, 1])

        w_position_encoder = self._get_sinusoid_encoding_table(n_width, d_hid)
        w_position_encoder = w_position_encoder.transpose([1, 0])
        w_position_encoder = w_position_encoder.reshape([1, d_hid, 1, n_width])

        self.register_buffer('h_position_encoder', h_position_encoder)
        self.register_buffer('w_position_encoder', w_position_encoder)

        self.h_scale = self.scale_factor_generate(d_hid)
        self.w_scale = self.scale_factor_generate(d_hid)
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.dropout = nn.Dropout(p=dropout)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        denominator = paddle.to_tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        denominator = denominator.reshape([1, -1])
        pos_tensor = paddle.cast(
            paddle.arange(n_position).unsqueeze(-1), 'float32')
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = paddle.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = paddle.cos(sinusoid_table[:, 1::2])

        return sinusoid_table

    def scale_factor_generate(self, d_hid):
        scale_factor = nn.Sequential(
            nn.Conv2D(d_hid, d_hid, 1),
            nn.ReLU(), nn.Conv2D(d_hid, d_hid, 1), nn.Sigmoid())

        return scale_factor

    def forward(self, x):
        b, c, h, w = x.shape

        avg_pool = self.pool(x)

        h_pos_encoding = \
            self.h_scale(avg_pool) * self.h_position_encoder[:, :, :h, :]
        w_pos_encoding = \
            self.w_scale(avg_pool) * self.w_position_encoder[:, :, :, :w]

        out = x + h_pos_encoding + w_pos_encoding

        out = self.dropout(out)

        return out


class ScaledDotProductAttention(nn.Layer):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        def masked_fill(x, mask, value):
            y = paddle.full(x.shape, value, x.dtype)
            return paddle.where(mask, y, x)

        attn = paddle.matmul(q / self.temperature, k.transpose([0, 1, 3, 2]))
        if mask is not None:
            attn = masked_fill(attn, mask == 0, -1e9)
            # attn = attn.masked_fill(mask == 0, float('-inf'))
            # attn += mask

        attn = self.dropout(F.softmax(attn, axis=-1))
        output = paddle.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Layer):
    def __init__(self,
                 n_head=8,
                 d_model=512,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v

        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias_attr=qkv_bias)
        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias_attr=qkv_bias)
        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias_attr=qkv_bias)

        self.attention = ScaledDotProductAttention(d_k**0.5, dropout)

        self.fc = nn.Linear(self.dim_v, d_model, bias_attr=qkv_bias)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, _ = q.shape
        _, len_k, _ = k.shape

        q = self.linear_q(q).reshape([batch_size, len_q, self.n_head, self.d_k])
        k = self.linear_k(k).reshape([batch_size, len_k, self.n_head, self.d_k])
        v = self.linear_v(v).reshape([batch_size, len_k, self.n_head, self.d_v])

        q, k, v = q.transpose([0, 2, 1, 3]), k.transpose(
            [0, 2, 1, 3]), v.transpose([0, 2, 1, 3])

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)

        attn_out, _ = self.attention(q, k, v, mask=mask)

        attn_out = attn_out.transpose([0, 2, 1, 3]).reshape(
            [batch_size, len_q, self.dim_v])

        attn_out = self.fc(attn_out)
        attn_out = self.proj_drop(attn_out)

        return attn_out


class SATRNEncoder(nn.Layer):
    def __init__(self,
                 n_layers=12,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 n_position=100,
                 d_inner=256,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.position_enc = Adaptive2DPositionalEncoding(
            d_hid=d_model,
            n_height=n_position,
            n_width=n_position,
            dropout=dropout)
        self.layer_stack = nn.LayerList([
            SATRNEncoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, feat, valid_ratios=None):
        """
        Args:
            feat (Tensor): Feature tensor of shape :math:`(N, D_m, H, W)`.
            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.

        Returns:
            Tensor: A tensor of shape :math:`(N, T, D_m)`.
        """
        if valid_ratios is None:
            valid_ratios = [1.0 for _ in range(feat.shape[0])]
        feat = self.position_enc(feat)
        n, c, h, w = feat.shape

        mask = paddle.zeros((n, h, w))
        for i, valid_ratio in enumerate(valid_ratios):
            valid_width = min(w, math.ceil(w * valid_ratio))
            mask[i, :, :valid_width] = 1

        mask = mask.reshape([n, h * w])
        feat = feat.reshape([n, c, h * w])

        output = feat.transpose([0, 2, 1])
        for enc_layer in self.layer_stack:
            output = enc_layer(output, h, w, mask)
        output = self.layer_norm(output)

        return output


class PositionwiseFeedForward(nn.Layer):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x


class PositionalEncoding(nn.Layer):
    def __init__(self, d_hid=512, n_position=200, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Not a parameter
        # Position table of shape (1, n_position, d_hid)
        self.register_buffer(
            'position_table',
            self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        denominator = paddle.to_tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        denominator = denominator.reshape([1, -1])
        pos_tensor = paddle.cast(
            paddle.arange(n_position).unsqueeze(-1), 'float32')
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = paddle.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = paddle.cos(sinusoid_table[:, 1::2])

        return sinusoid_table.unsqueeze(0)

    def forward(self, x):

        x = x + self.position_table[:, :x.shape[1]].clone().detach()
        return self.dropout(x)


class TFDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 operation_order=None):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)

        self.enc_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)

        self.mlp = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.operation_order = operation_order
        if self.operation_order is None:
            self.operation_order = ('norm', 'self_attn', 'norm', 'enc_dec_attn',
                                    'norm', 'ffn')
        assert self.operation_order in [
            ('norm', 'self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn'),
            ('self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn', 'norm')
        ]

    def forward(self,
                dec_input,
                enc_output,
                self_attn_mask=None,
                dec_enc_attn_mask=None):
        if self.operation_order == ('self_attn', 'norm', 'enc_dec_attn', 'norm',
                                    'ffn', 'norm'):
            dec_attn_out = self.self_attn(dec_input, dec_input, dec_input,
                                          self_attn_mask)
            dec_attn_out += dec_input
            dec_attn_out = self.norm1(dec_attn_out)

            enc_dec_attn_out = self.enc_attn(dec_attn_out, enc_output,
                                             enc_output, dec_enc_attn_mask)
            enc_dec_attn_out += dec_attn_out
            enc_dec_attn_out = self.norm2(enc_dec_attn_out)

            mlp_out = self.mlp(enc_dec_attn_out)
            mlp_out += enc_dec_attn_out
            mlp_out = self.norm3(mlp_out)
        elif self.operation_order == ('norm', 'self_attn', 'norm',
                                      'enc_dec_attn', 'norm', 'ffn'):
            dec_input_norm = self.norm1(dec_input)
            dec_attn_out = self.self_attn(dec_input_norm, dec_input_norm,
                                          dec_input_norm, self_attn_mask)
            dec_attn_out += dec_input

            enc_dec_attn_in = self.norm2(dec_attn_out)
            enc_dec_attn_out = self.enc_attn(enc_dec_attn_in, enc_output,
                                             enc_output, dec_enc_attn_mask)
            enc_dec_attn_out += dec_attn_out

            mlp_out = self.mlp(self.norm3(enc_dec_attn_out))
            mlp_out += enc_dec_attn_out

        return mlp_out


class SATRNDecoder(nn.Layer):
    def __init__(self,
                 n_layers=6,
                 d_embedding=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=256,
                 n_position=200,
                 dropout=0.1,
                 num_classes=93,
                 max_seq_len=40,
                 start_idx=1,
                 padding_idx=92):
        super().__init__()

        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.max_seq_len = max_seq_len

        self.trg_word_emb = nn.Embedding(
            num_classes, d_embedding, padding_idx=padding_idx)

        self.position_enc = PositionalEncoding(
            d_embedding, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.LayerList([
            TFDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, epsilon=1e-6)

        pred_num_class = num_classes - 1  # ignore padding_idx
        self.classifier = nn.Linear(d_model, pred_num_class)

    @staticmethod
    def get_pad_mask(seq, pad_idx):

        return (seq != pad_idx).unsqueeze(-2)

    @staticmethod
    def get_subsequent_mask(seq):
        """For masking out the subsequent info."""
        len_s = seq.shape[1]
        subsequent_mask = 1 - paddle.triu(
            paddle.ones((len_s, len_s)), diagonal=1)
        subsequent_mask = paddle.cast(subsequent_mask.unsqueeze(0), 'bool')

        return subsequent_mask

    def _attention(self, trg_seq, src, src_mask=None):
        trg_embedding = self.trg_word_emb(trg_seq)
        trg_pos_encoded = self.position_enc(trg_embedding)
        tgt = self.dropout(trg_pos_encoded)

        trg_mask = self.get_pad_mask(
            trg_seq,
            pad_idx=self.padding_idx) & self.get_subsequent_mask(trg_seq)
        output = tgt
        for dec_layer in self.layer_stack:
            output = dec_layer(
                output,
                src,
                self_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask)
        output = self.layer_norm(output)

        return output

    def _get_mask(self, logit, valid_ratios):
        N, T, _ = logit.shape
        mask = None
        if valid_ratios is not None:
            mask = paddle.zeros((N, T))
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(T, math.ceil(T * valid_ratio))
                mask[i, :valid_width] = 1

        return mask

    def forward_train(self, feat, out_enc, targets, valid_ratio):
        src_mask = self._get_mask(out_enc, valid_ratio)
        attn_output = self._attention(targets, out_enc, src_mask=src_mask)
        outputs = self.classifier(attn_output)

        return outputs

    def forward_test(self, feat, out_enc, valid_ratio):

        src_mask = self._get_mask(out_enc, valid_ratio)
        N = out_enc.shape[0]
        init_target_seq = paddle.full(
            (N, self.max_seq_len + 1), self.padding_idx, dtype='int64')
        # bsz * seq_len
        init_target_seq[:, 0] = self.start_idx

        outputs = []
        for step in range(0, paddle.to_tensor(self.max_seq_len)):
            decoder_output = self._attention(
                init_target_seq, out_enc, src_mask=src_mask)
            # bsz * seq_len * C
            step_result = F.softmax(
                self.classifier(decoder_output[:, step, :]), axis=-1)
            # bsz * num_classes
            outputs.append(step_result)
            step_max_index = paddle.argmax(step_result, axis=-1)
            init_target_seq[:, step + 1] = step_max_index

        outputs = paddle.stack(outputs, axis=1)

        return outputs

    def forward(self, feat, out_enc, targets=None, valid_ratio=None):
        if self.training:
            return self.forward_train(feat, out_enc, targets, valid_ratio)
        else:
            return self.forward_test(feat, out_enc, valid_ratio)


class SATRNHead(nn.Layer):
    def __init__(self, enc_cfg, dec_cfg, **kwargs):
        super(SATRNHead, self).__init__()

        # encoder module
        self.encoder = SATRNEncoder(**enc_cfg)

        # decoder module
        self.decoder = SATRNDecoder(**dec_cfg)

    def forward(self, feat, targets=None):

        if targets is not None:
            targets, valid_ratio = targets
        else:
            targets, valid_ratio = None, None
        holistic_feat = self.encoder(feat, valid_ratio)  # bsz c

        final_out = self.decoder(feat, holistic_feat, targets, valid_ratio)

        return final_out
