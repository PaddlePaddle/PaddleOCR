# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/mmocr/models/textrecog/decoders/master_decoder.py
"""

import copy
import math
import paddle
from paddle import nn
from paddle.nn import functional as F


class TableMasterHead(nn.Layer):
    """
    Split to two transformer header at the last layer.
    Cls_layer is used to structure token classification.
    Bbox_layer is used to regress bbox coord.
    """

    def __init__(self,
                 in_channels,
                 out_channels=30,
                 headers=8,
                 d_ff=2048,
                 dropout=0,
                 max_text_length=500,
                 loc_reg_num=4,
                 **kwargs):
        super(TableMasterHead, self).__init__()
        hidden_size = in_channels[-1]
        self.layers = clones(
            DecoderLayer(headers, hidden_size, dropout, d_ff), 2)
        self.cls_layer = clones(
            DecoderLayer(headers, hidden_size, dropout, d_ff), 1)
        self.bbox_layer = clones(
            DecoderLayer(headers, hidden_size, dropout, d_ff), 1)
        self.cls_fc = nn.Linear(hidden_size, out_channels)
        self.bbox_fc = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, loc_reg_num),
            nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_size)
        self.embedding = Embeddings(d_model=hidden_size, vocab=out_channels)
        self.positional_encoding = PositionalEncoding(d_model=hidden_size)

        self.SOS = out_channels - 3
        self.PAD = out_channels - 1
        self.out_channels = out_channels
        self.loc_reg_num = loc_reg_num
        self.max_text_length = max_text_length

    def make_mask(self, tgt):
        """
        Make mask for self attention.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3)

        tgt_len = paddle.shape(tgt)[1]
        trg_sub_mask = paddle.tril(
            paddle.ones(
                ([tgt_len, tgt_len]), dtype=paddle.float32))

        tgt_mask = paddle.logical_and(
            trg_pad_mask.astype(paddle.float32), trg_sub_mask)
        return tgt_mask.astype(paddle.float32)

    def decode(self, input, feature, src_mask, tgt_mask):
        # main process of transformer decoder.
        x = self.embedding(input)  # x: 1*x*512, feature: 1*3600,512
        x = self.positional_encoding(x)

        # origin transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, feature, src_mask, tgt_mask)

        # cls head
        cls_x = x
        for layer in self.cls_layer:
            cls_x = layer(x, feature, src_mask, tgt_mask)
        cls_x = self.norm(cls_x)

        # bbox head
        bbox_x = x
        for layer in self.bbox_layer:
            bbox_x = layer(x, feature, src_mask, tgt_mask)
        bbox_x = self.norm(bbox_x)
        return self.cls_fc(cls_x), self.bbox_fc(bbox_x)

    def greedy_forward(self, SOS, feature):
        input = SOS
        output = paddle.zeros(
            [input.shape[0], self.max_text_length + 1, self.out_channels])
        bbox_output = paddle.zeros(
            [input.shape[0], self.max_text_length + 1, self.loc_reg_num])
        max_text_length = paddle.to_tensor(self.max_text_length)
        for i in range(max_text_length + 1):
            target_mask = self.make_mask(input)
            out_step, bbox_output_step = self.decode(input, feature, None,
                                                     target_mask)
            prob = F.softmax(out_step, axis=-1)
            next_word = prob.argmax(axis=2, dtype="int64")
            input = paddle.concat(
                [input, next_word[:, -1].unsqueeze(-1)], axis=1)
            if i == self.max_text_length:
                output = out_step
                bbox_output = bbox_output_step
        return output, bbox_output

    def forward_train(self, out_enc, targets):
        # x is token of label
        # feat is feature after backbone before pe.
        # out_enc is feature after pe.
        padded_targets = targets[0]
        src_mask = None
        tgt_mask = self.make_mask(padded_targets[:, :-1])
        output, bbox_output = self.decode(padded_targets[:, :-1], out_enc,
                                          src_mask, tgt_mask)
        return {'structure_probs': output, 'loc_preds': bbox_output}

    def forward_test(self, out_enc):
        batch_size = out_enc.shape[0]
        SOS = paddle.zeros([batch_size, 1], dtype='int64') + self.SOS
        output, bbox_output = self.greedy_forward(SOS, out_enc)
        output = F.softmax(output)
        return {'structure_probs': output, 'loc_preds': bbox_output}

    def forward(self, feat, targets=None):
        feat = feat[-1]
        b, c, h, w = feat.shape
        feat = feat.reshape([b, c, h * w])  # flatten 2D feature map
        feat = feat.transpose((0, 2, 1))
        out_enc = self.positional_encoding(feat)
        if self.training:
            return self.forward_train(out_enc, targets)

        return self.forward_test(out_enc)


class DecoderLayer(nn.Layer):
    """
    Decoder is made of self attention, srouce attention and feed forward.
    """

    def __init__(self, headers, d_model, dropout, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(headers, d_model, dropout)
        self.src_attn = MultiHeadAttention(headers, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SubLayerConnection(d_model, dropout), 3)

    def forward(self, x, feature, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](
            x, lambda x: self.src_attn(x, feature, feature, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadAttention(nn.Layer):
    def __init__(self, headers, d_model, dropout):
        super(MultiHeadAttention, self).__init__()

        assert d_model % headers == 0
        self.d_k = int(d_model / headers)
        self.headers = headers
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.shape[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).reshape([B, 0, self.headers, self.d_k]).transpose([0, 2, 1, 3])
             for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = self_attention(
            query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose([0, 2, 1, 3]).reshape([B, 0, self.headers * self.d_k])
        return self.linears[-1](x)


class FeedForward(nn.Layer):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SubLayerConnection(nn.Layer):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def masked_fill(x, mask, value):
    mask = mask.astype(x.dtype)
    return x * paddle.logical_not(mask).astype(x.dtype) + mask * value


def self_attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scale Dot Product Attention'
    """
    d_k = value.shape[-1]

    score = paddle.matmul(query, key.transpose([0, 1, 3, 2]) / math.sqrt(d_k))
    if mask is not None:
        # score = score.masked_fill(mask == 0, -1e9) # b, h, L, L
        score = masked_fill(score, mask == 0, -6.55e4)  # for fp16

    p_attn = F.softmax(score, axis=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return paddle.matmul(p_attn, value), p_attn


def clones(module, N):
    """ Produce N identical layers """
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Layer):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input):
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Layer):
    """ Implement the PE function. """

    def __init__(self, d_model, dropout=0., max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = paddle.zeros([max_len, d_model])
        position = paddle.arange(0, max_len).unsqueeze(1).astype('float32')
        div_term = paddle.exp(
            paddle.arange(0, d_model, 2) * -math.log(10000.0) / d_model)
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, feat, **kwargs):
        feat = feat + self.pe[:, :paddle.shape(feat)[1]]  # pe 1*5000*512
        return self.dropout(feat)
