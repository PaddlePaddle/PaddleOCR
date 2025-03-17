# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

# Code was based on https://github.com/baudm/parseq/blob/main/strhub/models/parseq/system.py
# reference: https://arxiv.org/abs/2207.06966

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import numpy as np
from .self_attention import WrapEncoderForFeature
from .self_attention import WrapEncoder
from collections import OrderedDict
from typing import Optional
import copy
from itertools import permutations


class DecoderLayer(paddle.nn.Layer):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
    This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-05,
    ):
        super().__init__()
        self.self_attn = paddle.nn.MultiHeadAttention(
            d_model, nhead, dropout=dropout, need_weights=True
        )  # paddle.nn.MultiHeadAttention默认为batch_first模式
        self.cross_attn = paddle.nn.MultiHeadAttention(
            d_model, nhead, dropout=dropout, need_weights=True
        )
        self.linear1 = paddle.nn.Linear(
            in_features=d_model, out_features=dim_feedforward
        )
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.linear2 = paddle.nn.Linear(
            in_features=dim_feedforward, out_features=d_model
        )
        self.norm1 = paddle.nn.LayerNorm(
            normalized_shape=d_model, epsilon=layer_norm_eps
        )
        self.norm2 = paddle.nn.LayerNorm(
            normalized_shape=d_model, epsilon=layer_norm_eps
        )
        self.norm_q = paddle.nn.LayerNorm(
            normalized_shape=d_model, epsilon=layer_norm_eps
        )
        self.norm_c = paddle.nn.LayerNorm(
            normalized_shape=d_model, epsilon=layer_norm_eps
        )
        self.dropout1 = paddle.nn.Dropout(p=dropout)
        self.dropout2 = paddle.nn.Dropout(p=dropout)
        self.dropout3 = paddle.nn.Dropout(p=dropout)
        if activation == "gelu":
            self.activation = paddle.nn.GELU()

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = paddle.nn.functional.gelu
        super().__setstate__(state)

    def forward_stream(
        self, tgt, tgt_norm, tgt_kv, memory, tgt_mask, tgt_key_padding_mask
    ):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        if tgt_key_padding_mask is not None:
            tgt_mask1 = (tgt_mask != float("-inf"))[None, None, :, :] & (
                tgt_key_padding_mask[:, None, None, :] == False
            )
            tgt2, sa_weights = self.self_attn(
                tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask1
            )
        else:
            tgt2, sa_weights = self.self_attn(
                tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask
            )

        tgt = tgt + self.dropout1(tgt2)
        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(
            self.dropout(self.activation(self.linear1(self.norm2(tgt))))
        )
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(
        self,
        query,
        content,
        memory,
        query_mask=None,
        content_mask=None,
        content_key_padding_mask=None,
        update_content=True,
    ):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(
            query,
            query_norm,
            content_norm,
            memory,
            query_mask,
            content_key_padding_mask,
        )[0]
        if update_content:
            content = self.forward_stream(
                content,
                content_norm,
                content_norm,
                memory,
                content_mask,
                content_key_padding_mask,
            )[0]
        return query, content


def get_clones(module, N):
    return paddle.nn.LayerList([copy.deepcopy(module) for i in range(N)])


class Decoder(paddle.nn.Layer):
    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        query,
        content,
        memory,
        query_mask: Optional[paddle.Tensor] = None,
        content_mask: Optional[paddle.Tensor] = None,
        content_key_padding_mask: Optional[paddle.Tensor] = None,
    ):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(
                query,
                content,
                memory,
                query_mask,
                content_mask,
                content_key_padding_mask,
                update_content=not last,
            )
        query = self.norm(query)
        return query


class TokenEmbedding(paddle.nn.Layer):
    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = paddle.nn.Embedding(
            num_embeddings=charset_size, embedding_dim=embed_dim
        )
        self.embed_dim = embed_dim

    def forward(self, tokens: paddle.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens.astype(paddle.int64))


def trunc_normal_init(param, **kwargs):
    initializer = nn.initializer.TruncatedNormal(**kwargs)
    initializer(param, param.block)


def constant_init(param, **kwargs):
    initializer = nn.initializer.Constant(**kwargs)
    initializer(param, param.block)


def kaiming_normal_init(param, **kwargs):
    initializer = nn.initializer.KaimingNormal(**kwargs)
    initializer(param, param.block)


class ParseQHead(nn.Layer):
    def __init__(
        self,
        out_channels,
        max_text_length,
        embed_dim,
        dec_num_heads,
        dec_mlp_ratio,
        dec_depth,
        perm_num,
        perm_forward,
        perm_mirrored,
        decode_ar,
        refine_iters,
        dropout,
        **kwargs,
    ):
        super().__init__()

        self.bos_id = out_channels - 2
        self.eos_id = 0
        self.pad_id = out_channels - 1

        self.max_label_length = max_text_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters
        decoder_layer = DecoderLayer(
            embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout
        )
        self.decoder = Decoder(
            decoder_layer,
            num_layers=dec_depth,
            norm=paddle.nn.LayerNorm(normalized_shape=embed_dim),
        )
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored
        self.head = paddle.nn.Linear(
            in_features=embed_dim, out_features=out_channels - 2
        )
        self.text_embed = TokenEmbedding(out_channels, embed_dim)
        self.pos_queries = paddle.create_parameter(
            shape=paddle.empty(shape=[1, max_text_length + 1, embed_dim]).shape,
            dtype=paddle.empty(shape=[1, max_text_length + 1, embed_dim]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(shape=[1, max_text_length + 1, embed_dim])
            ),
        )
        self.pos_queries.stop_gradient = not True
        self.dropout = paddle.nn.Dropout(p=dropout)
        self._device = self.parameters()[0].place
        trunc_normal_init(self.pos_queries, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            trunc_normal_init(m.weight, std=0.02)
            if m.bias is not None:
                constant_init(m.bias, value=0.0)
        elif isinstance(m, paddle.nn.Embedding):
            trunc_normal_init(m.weight, std=0.02)
            if m._padding_idx is not None:
                m.weight.data[m._padding_idx].zero_()
        elif isinstance(m, paddle.nn.Conv2D):
            kaiming_normal_init(m.weight, fan_in=None, nonlinearity="relu")
            if m.bias is not None:
                constant_init(m.bias, value=0.0)
        elif isinstance(
            m, (paddle.nn.LayerNorm, paddle.nn.BatchNorm2D, paddle.nn.GroupNorm)
        ):
            constant_init(m.weight, value=1.0)
            constant_init(m.bias, value=0.0)

    def no_weight_decay(self):
        param_names = {"text_embed.embedding.weight", "pos_queries"}
        enc_param_names = {("encoder." + n) for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def encode(self, img):
        return self.encoder(img)

    def decode(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_padding_mask=None,
        tgt_query=None,
        tgt_query_mask=None,
    ):
        N, L = tgt.shape
        null_ctx = self.text_embed(tgt[:, :1])
        if L != 1:
            tgt_emb = self.pos_queries[:, : L - 1] + self.text_embed(tgt[:, 1:])
            tgt_emb = self.dropout(paddle.concat(x=[null_ctx, tgt_emb], axis=1))
        else:
            tgt_emb = self.dropout(null_ctx)
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(shape=[N, -1, -1])
        tgt_query = self.dropout(tgt_query)
        return self.decoder(
            tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask
        )

    def forward_test(self, memory, max_length=None):
        testing = max_length is None
        max_length = (
            self.max_label_length
            if max_length is None
            else min(max_length, self.max_label_length)
        )
        bs = memory.shape[0]
        num_steps = max_length + 1

        pos_queries = self.pos_queries[:, :num_steps].expand(shape=[bs, -1, -1])
        tgt_mask = query_mask = paddle.triu(
            x=paddle.full(shape=(num_steps, num_steps), fill_value=float("-inf")),
            diagonal=1,
        )
        if self.decode_ar:
            tgt_in = paddle.full(shape=(bs, num_steps), fill_value=self.pad_id).astype(
                "int64"
            )
            tgt_in[:, (0)] = self.bos_id

            logits = []
            for i in range(paddle.to_tensor(num_steps)):
                j = i + 1
                tgt_out = self.decode(
                    tgt_in[:, :j],
                    memory,
                    tgt_mask[:j, :j],
                    tgt_query=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                )
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    tgt_in[:, (j)] = p_i.squeeze().argmax(axis=-1)
                    if (
                        testing
                        and (tgt_in == self.eos_id)
                        .astype("bool")
                        .any(axis=-1)
                        .astype("bool")
                        .all()
                    ):
                        break
            logits = paddle.concat(x=logits, axis=1)
        else:
            tgt_in = paddle.full(shape=(bs, 1), fill_value=self.bos_id).astype("int64")
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)
        if self.refine_iters:
            temp = paddle.triu(
                x=paddle.ones(shape=[num_steps, num_steps], dtype="bool"), diagonal=2
            )
            posi = np.where(temp.cpu().numpy() == True)
            query_mask[posi] = 0
            bos = paddle.full(shape=(bs, 1), fill_value=self.bos_id).astype("int64")
            for i in range(self.refine_iters):
                tgt_in = paddle.concat(x=[bos, logits[:, :-1].argmax(axis=-1)], axis=1)
                tgt_padding_mask = (tgt_in == self.eos_id).astype(dtype="int32")
                tgt_padding_mask = tgt_padding_mask.cpu()
                tgt_padding_mask = tgt_padding_mask.cumsum(axis=-1) > 0
                tgt_padding_mask = (
                    tgt_padding_mask.cuda().astype(dtype="float32") == 1.0
                )
                tgt_out = self.decode(
                    tgt_in,
                    memory,
                    tgt_mask,
                    tgt_padding_mask,
                    tgt_query=pos_queries,
                    tgt_query_mask=query_mask[:, : tgt_in.shape[1]],
                )
                logits = self.head(tgt_out)

        # transfer to probability
        logits = F.softmax(logits, axis=-1)

        final_output = {"predict": logits}

        return final_output

    def gen_tgt_perms(self, tgt):
        """Generate shared permutations for the whole batch.
        This works because the same attention mask can be used for the shorter sequences
        because of the padding mask.
        """
        max_num_chars = tgt.shape[1] - 2
        if max_num_chars == 1:
            return paddle.arange(end=3).unsqueeze(axis=0)
        perms = [paddle.arange(end=max_num_chars)] if self.perm_forward else []
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        if max_num_chars < 5:
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = paddle.to_tensor(
                data=list(permutations(range(max_num_chars), max_num_chars)),
                place=self._device,
            )[selector]
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = paddle.stack(x=perms)
            if len(perm_pool):
                i = self.rng.choice(
                    len(perm_pool), size=num_gen_perms - len(perms), replace=False
                )
                perms = paddle.concat(x=[perms, perm_pool[i]])
        else:
            perms.extend(
                [
                    paddle.randperm(n=max_num_chars)
                    for _ in range(num_gen_perms - len(perms))
                ]
            )
            perms = paddle.stack(x=perms)
        if self.perm_mirrored:
            comp = perms.flip(axis=-1)
            x = paddle.stack(x=[perms, comp])
            perm_2 = list(range(x.ndim))
            perm_2[0] = 1
            perm_2[1] = 0
            perms = x.transpose(perm=perm_2).reshape((-1, max_num_chars))
        bos_idx = paddle.zeros(shape=(len(perms), 1), dtype=perms.dtype)
        eos_idx = paddle.full(
            shape=(len(perms), 1), fill_value=max_num_chars + 1, dtype=perms.dtype
        )
        perms = paddle.concat(x=[bos_idx, perms + 1, eos_idx], axis=1)
        if len(perms) > 1:
            perms[(1), 1:] = max_num_chars + 1 - paddle.arange(end=max_num_chars + 1)
        return perms

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = paddle.zeros(shape=(sz, sz))
        for i in range(sz):
            query_idx = perm[i].cpu().numpy().tolist()
            masked_keys = perm[i + 1 :].cpu().numpy().tolist()
            if len(masked_keys) == 0:
                break
            mask[query_idx, masked_keys] = float("-inf")
        content_mask = mask[:-1, :-1].clone()
        mask[paddle.eye(num_rows=sz).astype("bool")] = float("-inf")
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def forward_train(self, memory, tgt):
        tgt_perms = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
        logits_list = []
        final_out = {}
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            out = self.decode(
                tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask
            )
            logits = self.head(out)
            if i == 0:
                final_out["predict"] = logits
            logits = logits.flatten(stop_axis=1)
            logits_list.append(logits)

        final_out["logits_list"] = logits_list
        final_out["pad_id"] = self.pad_id
        final_out["eos_id"] = self.eos_id

        return final_out

    def forward(self, feat, targets=None):
        # feat : B, N, C
        # targets : labels, labels_len

        if self.training:
            label = targets[0]  # label
            label_len = targets[1]
            max_step = paddle.max(label_len).cpu().numpy()[0] + 2
            crop_label = label[:, :max_step]
            final_out = self.forward_train(feat, crop_label)
        else:
            final_out = self.forward_test(feat)

        return final_out
