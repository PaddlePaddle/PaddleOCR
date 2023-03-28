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
https://github.com/FangShancheng/ABINet/tree/main/modules
"""

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.nn import LayerList
from ppocr.modeling.heads.rec_nrtr_head import TransformerBlock, PositionalEncoding


class BCNLanguage(nn.Layer):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=2048,
                 dropout=0.,
                 max_length=25,
                 detach=True,
                 num_classes=37):
        super().__init__()

        self.d_model = d_model
        self.detach = detach
        self.max_length = max_length + 1  # additional stop token
        self.proj = nn.Linear(num_classes, d_model, bias_attr=False)
        self.token_encoder = PositionalEncoding(
            dropout=0.1, dim=d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(
            dropout=0, dim=d_model, max_len=self.max_length)

        self.decoder = nn.LayerList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=dropout,
                residual_dropout_rate=dropout,
                with_self_attn=False,
                with_cross_attn=True) for i in range(num_layers)
        ])

        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (B, N, C) where N is length, B is batch size and C is classes number
            lengths: (B,)
        """
        if self.detach: tokens = tokens.detach()
        embed = self.proj(tokens)  # (B, N, C)
        embed = self.token_encoder(embed)  # (B, N, C)
        padding_mask = _get_mask(lengths, self.max_length)
        zeros = paddle.zeros_like(embed)  # (B, N, C)
        qeury = self.pos_encoder(zeros)
        for decoder_layer in self.decoder:
            qeury = decoder_layer(qeury, embed, cross_mask=padding_mask)
        output = qeury  # (B, N, C)

        logits = self.cls(output)  # (B, N, C)

        return output, logits


def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(
        nn.Conv2D(in_c, out_c, k, s, p), nn.BatchNorm2D(out_c), nn.ReLU())


def decoder_layer(in_c,
                  out_c,
                  k=3,
                  s=1,
                  p=1,
                  mode='nearest',
                  scale_factor=None,
                  size=None):
    align_corners = False if mode == 'nearest' else True
    return nn.Sequential(
        nn.Upsample(
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners),
        nn.Conv2D(in_c, out_c, k, s, p),
        nn.BatchNorm2D(out_c),
        nn.ReLU())


class PositionAttention(nn.Layer):
    def __init__(self,
                 max_length,
                 in_channels=512,
                 num_channels=64,
                 h=8,
                 w=32,
                 mode='nearest',
                 **kwargs):
        super().__init__()
        self.max_length = max_length
        self.k_encoder = nn.Sequential(
            encoder_layer(
                in_channels, num_channels, s=(1, 2)),
            encoder_layer(
                num_channels, num_channels, s=(2, 2)),
            encoder_layer(
                num_channels, num_channels, s=(2, 2)),
            encoder_layer(
                num_channels, num_channels, s=(2, 2)))
        self.k_decoder = nn.Sequential(
            decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(
                num_channels, in_channels, size=(h, w), mode=mode))

        self.pos_encoder = PositionalEncoding(
            dropout=0, dim=in_channels, max_len=max_length)
        self.project = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        k, v = x, x

        # calculate key vector
        features = []
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(0, len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            # print(k.shape, features[len(self.k_decoder) - 2 - i].shape)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)

        # calculate query vector
        # TODO q=f(q,k)
        zeros = paddle.zeros(
            (B, self.max_length, C), dtype=x.dtype)  # (T, N, C)
        q = self.pos_encoder(zeros)  # (B, N, C)
        q = self.project(q)  # (B, N, C)

        # calculate attention
        attn_scores = q @k.flatten(2)  # (B, N, (H*W))
        attn_scores = attn_scores / (C**0.5)
        attn_scores = F.softmax(attn_scores, axis=-1)

        v = v.flatten(2).transpose([0, 2, 1])  # (B, (H*W), C)
        attn_vecs = attn_scores @v  # (B, N, C)

        return attn_vecs, attn_scores.reshape([0, self.max_length, H, W])


class ABINetHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 d_model=512,
                 nhead=8,
                 num_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_length=25,
                 use_lang=False,
                 iter_size=1):
        super().__init__()
        self.max_length = max_length + 1
        self.pos_encoder = PositionalEncoding(
            dropout=0.1, dim=d_model, max_len=8 * 32)
        self.encoder = nn.LayerList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=dropout,
                residual_dropout_rate=dropout,
                with_self_attn=True,
                with_cross_attn=False) for i in range(num_layers)
        ])
        self.decoder = PositionAttention(
            max_length=max_length + 1,  # additional stop token
            mode='nearest', )
        self.out_channels = out_channels
        self.cls = nn.Linear(d_model, self.out_channels)
        self.use_lang = use_lang
        if use_lang:
            self.iter_size = iter_size
            self.language = BCNLanguage(
                d_model=d_model,
                nhead=nhead,
                num_layers=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_length=max_length,
                num_classes=self.out_channels)
            # alignment
            self.w_att_align = nn.Linear(2 * d_model, d_model)
            self.cls_align = nn.Linear(d_model, self.out_channels)

    def forward(self, x, targets=None):
        x = x.transpose([0, 2, 3, 1])
        _, H, W, C = x.shape
        feature = x.flatten(1, 2)
        feature = self.pos_encoder(feature)
        for encoder_layer in self.encoder:
            feature = encoder_layer(feature)
        feature = feature.reshape([0, H, W, C]).transpose([0, 3, 1, 2])
        v_feature, attn_scores = self.decoder(
            feature)  # (B, N, C), (B, C, H, W)
        vis_logits = self.cls(v_feature)  # (B, N, C)
        logits = vis_logits
        vis_lengths = _get_length(vis_logits)
        if self.use_lang:
            align_logits = vis_logits
            align_lengths = vis_lengths
            all_l_res, all_a_res = [], []
            for i in range(self.iter_size):
                tokens = F.softmax(align_logits, axis=-1)
                lengths = align_lengths
                lengths = paddle.clip(
                    lengths, 2, self.max_length)  # TODO:move to langauge model
                l_feature, l_logits = self.language(tokens, lengths)

                # alignment
                all_l_res.append(l_logits)
                fuse = paddle.concat((l_feature, v_feature), -1)
                f_att = F.sigmoid(self.w_att_align(fuse))
                output = f_att * v_feature + (1 - f_att) * l_feature
                align_logits = self.cls_align(output)  # (B, N, C)

                align_lengths = _get_length(align_logits)
                all_a_res.append(align_logits)
            if self.training:
                return {
                    'align': all_a_res,
                    'lang': all_l_res,
                    'vision': vis_logits
                }
            else:
                logits = align_logits
        if self.training:
            return logits
        else:
            return F.softmax(logits, -1)


def _get_length(logit):
    """ Greed decoder to obtain length from logit"""
    out = (logit.argmax(-1) == 0)
    abn = out.any(-1)
    out_int = out.cast('int32')
    out = (out_int.cumsum(-1) == 1) & out
    out = out.cast('int32')
    out = out.argmax(-1)
    out = out + 1
    len_seq = paddle.zeros_like(out) + logit.shape[1]
    out = paddle.where(abn, out, len_seq)
    return out


def _get_mask(length, max_length):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    length = length.unsqueeze(-1)
    B = paddle.shape(length)[0]
    grid = paddle.arange(0, max_length).unsqueeze(0).tile([B, 1])
    zero_mask = paddle.zeros([B, max_length], dtype='float32')
    inf_mask = paddle.full([B, max_length], '-inf', dtype='float32')
    diag_mask = paddle.diag(
        paddle.full(
            [max_length], '-inf', dtype=paddle.float32),
        offset=0,
        name=None)
    mask = paddle.where(grid >= length, inf_mask, zero_mask)
    mask = mask.unsqueeze(1) + diag_mask
    return mask.unsqueeze(1)
