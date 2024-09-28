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
https://github.com/lukas-blecher/LaTeX-OCR/blob/main/pix2tex/models/transformer.py
"""

import math
import paddle
from paddle import nn, einsum
import paddle.nn.functional as F
from functools import partial
from inspect import isfunction
from collections import namedtuple

from paddle.nn.initializer import (
    TruncatedNormal,
    Constant,
    Normal,
    KaimingUniform,
    XavierUniform,
)

zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)
normal_ = Normal(std=0.02)
DEFAULT_DIM_HEAD = 64

Intermediates = namedtuple("Intermediates", ["pre_softmax_attn", "post_softmax_attn"])

LayerIntermediates = namedtuple("Intermediates", ["hiddens", "attn_intermediates"])

# helpers


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class always:
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class not_equals:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x != self.val


class equals:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x == self.val


def max_neg_value(tensor):
    return -paddle.finfo(tensor.dtype).max


def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(
        partial(string_begins_with, prefix), d
    )
    kwargs_without_prefix = dict(
        map(lambda x: (x[0][len(prefix) :], x[1]), tuple(kwargs_with_prefix.items()))
    )
    return kwargs_without_prefix, kwargs


# positional embeddings


class DepthWiseConv1d(nn.Layer):
    def __init__(
        self, dim_in, dim_out, kernel_size, padding=0, stride=1, bias=True, groups=False
    ):
        super().__init__()
        groups = default(groups, dim_in)
        self.net = nn.Sequential(
            nn.Conv1D(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                padding=padding,
                groups=dim_in,
                stride=stride,
                bias_attr=bias,
            ),
            nn.Conv1D(dim_in, dim_out, 1),
        )

    def forward(self, x):
        return self.net(x)


class AbsolutePositionalEmbedding(nn.Layer):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):

        normal_(self.emb.weight)

    def forward(self, x):
        n = paddle.arange(x.shape[1])
        return self.emb(n)[None, :, :]


class FixedPositionalEmbedding(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        t = (
            paddle.arange(
                x.shape[seq_dim],
            ).type_as(self.inv_freq)
            + offset
        )
        sinusoid_inp = paddle.einsum("i , j -> i j", t, self.inv_freq)
        emb = paddle.concat((sinusoid_inp.sin(), sinusoid_inp.cos()), axis=-1)
        return emb[None, :, :]


class Scale(nn.Layer):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.value, *rest)


class Rezero(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = paddle.create_parameter([1], dtype="float32")
        zeros_(self.g)

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.g, *rest)


class ScaleNorm(nn.Layer):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = paddle.create_parameter([1], dtype="float32")
        ones_(self.g)

    def forward(self, x):
        norm = paddle.norm(x, axis=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class RMSNorm(nn.Layer):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = paddle.create_parameter([dim])
        ones_(self.g)

    def forward(self, x):
        norm = paddle.norm(x, axis=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Residual(nn.Layer):
    def forward(self, x, residual):
        return x + residual


class GEGLU(nn.Layer):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, axis=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Layer):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Layer):
    def __init__(
        self,
        dim,
        dim_head=DEFAULT_DIM_HEAD,
        heads=8,
        causal=False,
        mask=None,
        talking_heads=False,
        collab_heads=False,
        collab_compression=0.3,
        sparse_topk=None,
        use_entmax15=False,
        num_mem_kv=0,
        dropout=0.0,
        on_attn=False,
        gate_values=False,
        is_export=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask
        self.is_export = is_export

        qk_dim = v_dim = dim_head * heads

        # collaborative heads
        self.collab_heads = collab_heads
        if self.collab_heads:
            qk_dim = int(collab_compression * qk_dim)
            self.collab_mixing = nn.Parameter(paddle.randn(heads, qk_dim))

        self.to_q = nn.Linear(dim, qk_dim, bias_attr=False)
        self.to_k = nn.Linear(dim, qk_dim, bias_attr=False)
        self.to_v = nn.Linear(dim, v_dim, bias_attr=False)

        self.dropout = nn.Dropout(dropout)

        # add GLU gating for aggregated values, from alphafold2
        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(dim, v_dim)
            zeros_(self.to_v_gate.weight)
            ones_(self.to_v_gate.bias)

        # talking heads
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_proj = nn.Parameter(paddle.randn(heads, heads))
            self.post_softmax_proj = nn.Parameter(paddle.randn(heads, heads))

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        self.attn_fn = F.softmax

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(paddle.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(paddle.randn(heads, num_mem_kv, dim_head))

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = (
            nn.Sequential(nn.Linear(v_dim, dim * 2), nn.GLU())
            if on_attn
            else nn.Linear(v_dim, dim)
        )

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        rel_pos=None,
        sinusoidal_emb=None,
        rotary_pos_emb=None,
        prev_attn=None,
        mem=None,
        seq_len=0,
    ):
        if not self.training:
            self.is_export = True
        b, n, _, h, talking_heads, collab_heads, has_context = (
            *x.shape,
            self.heads,
            self.talking_heads,
            self.collab_heads,
            exists(context),
        )
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        if exists(mem):
            k_input = paddle.concat((mem, k_input), axis=-2)
            v_input = paddle.concat((mem, v_input), axis=-2)

        if exists(sinusoidal_emb):
            # in shortformer, the query would start at a position offset depending on the past cached memory
            offset = k_input.shape[-2] - q_input.shape[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)
        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        def rearrange_q_k_v(x, h, is_export):
            if is_export:
                b, n, h_d = paddle.shape(x)
            else:
                b, n, h_d = x.shape
            d = h_d // h
            return x.reshape([b, n, h, d]).transpose([0, 2, 1, 3])

        q, k, v = map(
            lambda t: rearrange_q_k_v(t, h, is_export=self.is_export), (q, k, v)
        )

        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(
                mask,
                lambda: paddle.ones(
                    (b, n),
                ).cast(paddle.bool),
            )
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(
                k_mask, lambda: paddle.ones((b, k.shape[-2])).cast(paddle.bool)
            )

            q_mask = q_mask.reshape([q_mask.shape[0], 1, q_mask.shape[1], 1])
            k_mask = k_mask.reshape([k_mask.shape[0], 1, 1, k_mask.shape[1]])
            input_mask = q_mask * k_mask

        if collab_heads:
            k = k.expand(-1, h, -1, -1)
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        mask_value = max_neg_value(dots)

        if exists(prev_attn):
            dots = dots + prev_attn

        pre_softmax_attn = dots.clone()

        if talking_heads:
            dots = einsum(
                "b h i j, h k -> b k i j", dots, self.pre_softmax_proj
            ).contiguous()

        if exists(rel_pos):
            dots = rel_pos(dots)

        input_mask = input_mask.cast(paddle.bool)
        if exists(input_mask):

            dots.masked_fill_(~input_mask, mask_value)
            del input_mask

        if self.causal:
            i, j = dots.shape[-2:]
            r = paddle.arange(i)
            r_shape = r.shape[0]
            mask = r.reshape([1, 1, r_shape, 1]) < r.reshape([1, 1, 1, r_shape])

            if self.is_export:
                pad_list = [
                    paddle.to_tensor(0, dtype="int32"),
                    paddle.to_tensor(0, dtype="int32"),
                    paddle.to_tensor(j - i, dtype="int32"),
                    paddle.to_tensor(0, dtype="int32"),
                ]
                mask = F.pad(
                    mask.cast(paddle.int32),
                    paddle.to_tensor(pad_list).cast(paddle.int32),
                    value=False,
                ).cast(paddle.bool)
                dots = dots.masked_fill_(mask, mask_value)
            else:
                mask = F.pad(mask.cast(paddle.int32), (0, 0, j - i, 0), value=False)
                dots.masked_fill_(mask, mask_value)
            del mask
        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots.masked_fill_(mask, mask_value)
            del mask

        attn = self.attn_fn(dots, axis=-1)
        post_softmax_attn = attn.clone()

        attn = self.dropout(attn)

        if talking_heads:
            attn = einsum(
                "b h i j, h k -> b k i j", attn, self.post_softmax_proj
            ).contiguous()
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        b, h, n, d = out.shape
        out = out.transpose([0, 2, 1, 3]).reshape([b, n, h * d])

        if exists(self.to_v_gate):
            gates = self.gate_v(x)
            out = out * gates.sigmoid()

        intermediates = Intermediates(
            pre_softmax_attn=pre_softmax_attn, post_softmax_attn=post_softmax_attn
        )

        return self.to_out(out), intermediates


class AttentionLayers(nn.Layer):
    def __init__(
        self,
        dim,
        depth,
        heads=8,
        causal=False,
        cross_attend=False,
        only_cross=False,
        use_scalenorm=False,
        use_rmsnorm=False,
        use_rezero=False,
        rel_pos_bias=False,
        rel_pos_num_buckets=32,
        rel_pos_max_distance=128,
        position_infused_attn=False,
        rotary_pos_emb=False,
        rotary_emb_dim=None,
        custom_layers=None,
        sandwich_coef=None,
        par_ratio=None,
        residual_attn=False,
        cross_residual_attn=False,
        macaron=False,
        pre_norm=True,
        gate_residual=False,
        is_export=False,
        **kwargs,
    ):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim("ff_", kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim("attn_", kwargs)

        dim_head = attn_kwargs.get("dim_head", DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.layers = nn.LayerList([])

        self.has_pos_emb = position_infused_attn or rel_pos_bias or rotary_pos_emb
        self.pia_pos_emb = (
            FixedPositionalEmbedding(dim) if position_infused_attn else None
        )

        assert (
            rel_pos_num_buckets <= rel_pos_max_distance
        ), "number of relative position buckets must be less than the relative position max distance"

        self.pre_norm = pre_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        self.cross_attend = cross_attend
        self.rel_pos = None

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)

        norm_fn = nn.Identity if use_rezero else norm_fn
        branch_fn = Rezero if use_rezero else None

        if cross_attend and not only_cross:
            default_block = ("a", "c", "f")
        elif cross_attend and only_cross:
            default_block = ("c", "f")
        else:
            default_block = ("a", "f")
        if macaron:
            default_block = ("f",) + default_block

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, "par ratio out of range"
            default_block = tuple(filter(not_equals("f"), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = (
                par_depth * 2 // 3
            )  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert (
                len(default_block) <= par_width
            ), "default block is too large for par_ratio"
            par_block = default_block + ("f",) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ("f",) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert (
                sandwich_coef > 0 and sandwich_coef <= depth
            ), "sandwich coefficient should be less than the depth"
            layer_types = (
                ("a",) * sandwich_coef
                + default_block * (depth - sandwich_coef)
                + ("f",) * sandwich_coef
            )
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals("a"), layer_types)))
        for layer_type in self.layer_types:
            if layer_type == "a":
                layer = Attention(
                    dim, heads=heads, causal=causal, is_export=is_export, **attn_kwargs
                )
            elif layer_type == "c":
                layer = Attention(dim, heads=heads, is_export=is_export, **attn_kwargs)
            elif layer_type == "f":
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f"invalid layer type {layer_type}")
            if isinstance(layer, Attention) and exists(branch_fn):
                layer = branch_fn(layer)
            residual_fn = Residual()
            self.layers.append(nn.LayerList([norm_fn(), layer, residual_fn]))

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        mems=None,
        seq_len=0,
        return_hiddens=False,
    ):
        assert not (
            self.cross_attend ^ exists(context)
        ), "context must be passed in if cross_attend is set to True"

        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None
        rotary_pos_emb = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(
            zip(self.layer_types, self.layers)
        ):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == "a":
                hiddens.append(x)
                layer_mem = mems.pop(0)

            residual = x

            if self.pre_norm:
                x = norm(x)

            if layer_type == "a":
                out, inter = block(
                    x,
                    mask=mask,
                    sinusoidal_emb=self.pia_pos_emb,
                    rel_pos=self.rel_pos,
                    rotary_pos_emb=rotary_pos_emb,
                    prev_attn=prev_attn,
                    mem=layer_mem,
                )
            elif layer_type == "c":
                out, inter = block(
                    x,
                    context=context,
                    mask=mask,
                    context_mask=context_mask,
                    prev_attn=prev_cross_attn,
                )
            elif layer_type == "f":
                out = block(x)

            x = residual_fn(out, residual)

            if layer_type in ("a", "c"):
                intermediates.append(inter)

            if layer_type == "a" and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == "c" and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if not self.pre_norm and not is_last:
                x = norm(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens, attn_intermediates=intermediates
            )

            return x, intermediates

        return x


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on encoder"
        super().__init__(causal=False, **kwargs)


class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on decoder"
        super().__init__(causal=True, **kwargs)


class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend=True, only_cross=True, **kwargs)


def create_latex_parameter(shape):
    return paddle.create_parameter(
        shape=shape,
        dtype="float32",
        default_initializer=paddle.nn.initializer.Assign(paddle.randn(shape)),
    )


class TransformerDecoder(nn.Layer):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers,
        emb_dim=None,
        max_mem_len=0.0,
        emb_dropout=0.0,
        num_memory_tokens=None,
        tie_embedding=False,
        use_pos_emb=True,
        is_export=False,
    ):
        super().__init__()
        assert isinstance(
            attn_layers, AttentionLayers
        ), "attention layers must be one of Encoder or Decoder"

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = (
            AbsolutePositionalEmbedding(emb_dim, max_seq_len)
            if (use_pos_emb and not attn_layers.has_pos_emb)
            else always(0)
        )
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.is_export = is_export

        self.init_()

        self.to_logits = (
            nn.Linear(dim, num_tokens)
            if not tie_embedding
            else lambda t: t @ self.token_emb.weight.t()
        )

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = create_latex_parameter([num_memory_tokens, dim])

            # let funnel encoder know number of memory tokens, if specified
            # TODO: think of a cleaner solution
            if hasattr(attn_layers, "num_memory_tokens"):
                attn_layers.num_memory_tokens = num_memory_tokens

    def init_(self):
        normal_(self.token_emb.weight)

    def forward(
        self,
        x,
        return_embeddings=False,
        mask=None,
        return_mems=False,
        return_attn=False,
        seq_len=0,
        mems=None,
        **kwargs,
    ):
        b, n, num_mem = *x.shape, self.num_memory_tokens
        x = self.token_emb(x)
        x = x + self.pos_emb(x)

        x = self.emb_dropout(x)
        x = self.project_emb(x)

        x, intermediates = self.attn_layers(
            x, mask=mask, mems=mems, return_hiddens=True, seq_len=seq_len, **kwargs
        )
        x = self.norm(x)
        mem, x = x[:, :num_mem], x[:, num_mem:]
        out = self.to_logits(x) if not return_embeddings else x
        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = (
                list(map(lambda pair: paddle.concat(pair, axis=-2), zip(mems, hiddens)))
                if exists(mems)
                else hiddens
            )
            new_mems = list(
                map(lambda t: t[..., -self.max_mem_len :, :].detach(), new_mems)
            )
            return out, new_mems

        if return_attn:
            attn_maps = list(
                map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates)
            )
            return out, attn_maps

        return out


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = paddle.sort(logits, descending=True)
    cum_probs = paddle.cumsum(F.softmax(sorted_logits, axis=-1), axis=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


# topk


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = paddle.topk(logits, k)
    probs = paddle.full_like(logits, float("-inf"))
    probs = paddle.put_along_axis(probs, ind, val, 1)
    return probs


class LaTeXOCRHead(nn.Layer):
    """Implementation of LaTeX OCR decoder.

    Args:
      encoded_feat: The encoded features with shape[N, 1, H//16, W//16]
      tgt_seq: LaTeX-OCR labels with shape [N, L] , L is the max sequence length
      xi: The first N-1 LaTeX-OCR sequences in tgt_seq with shape [N, L-1]
      mask: The first N-1 LaTeX-OCR attention mask with shape [N, L-1]  , L is the max sequence length

    Returns:
      The predicted LaTeX sequences with shape [N, L-1, C], C is the number of LaTeX classes
    """

    def __init__(
        self,
        net=None,
        in_channels=256,
        out_channels=256,
        pad_value=0,
        decoder_args=None,
        is_export=False,
    ):
        super().__init__()
        decoder = Decoder(
            dim=256, depth=4, heads=8, is_export=is_export, **decoder_args
        )
        transformer_decoder = TransformerDecoder(
            num_tokens=8000,
            max_seq_len=512,
            attn_layers=decoder,
            is_export=is_export,
        )
        self.temperature = 0.333
        self.bos_token = 1
        self.eos_token = 2
        self.max_length = 512
        self.pad_value = pad_value

        self.net = transformer_decoder
        self.max_seq_len = self.net.max_seq_len
        self.is_export = is_export

    @paddle.no_grad()
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token=None,
        temperature=1.0,
        filter_logits_fn=top_k,
        filter_thres=0.9,
        **kwargs,
    ):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop("mask", None)

        if mask is None:
            mask = paddle.full_like(out, True, dtype=paddle.bool)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len :]
            mask = mask[:, -self.max_seq_len :]
            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]
            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)

                probs = F.softmax(filtered_logits / temperature, axis=-1)
            else:
                raise NotImplementedError("The filter_logits_fn is not supported ")

            sample = paddle.multinomial(probs, 1)
            out = paddle.concat((out, sample), axis=-1)
            pad_mask = paddle.full(shape=[mask.shape[0], 1], fill_value=1, dtype="bool")
            mask = paddle.concat((mask, pad_mask), axis=1)
            if (
                eos_token is not None
                and (
                    paddle.cumsum((out == eos_token).cast(paddle.int64), 1)[:, -1] >= 1
                ).all()
            ):
                break
        out = out[:, t:]
        if num_dims == 1:
            out = out.squeeze(0)
        return out

    @paddle.no_grad()
    def generate_export(
        self,
        start_tokens,
        seq_len,
        eos_token=None,
        context=None,
        temperature=1.0,
        filter_logits_fn=None,
        filter_thres=0.9,
        **kwargs,
    ):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop("mask", None)

        if mask is None:
            mask = paddle.full_like(out, True, dtype=paddle.bool)

        i_idx = paddle.full([], 0)
        while i_idx < paddle.to_tensor(seq_len):
            x = out[:, -self.max_seq_len :]
            paddle.jit.api.set_dynamic_shape(x, [-1, -1])
            mask = mask[:, -self.max_seq_len :]
            paddle.jit.api.set_dynamic_shape(mask, [-1, -1])
            logits = self.net(x, mask=mask, context=context, seq_len=i_idx, **kwargs)[
                :, -1, :
            ]
            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)

                probs = F.softmax(filtered_logits / temperature, axis=-1)

            sample = paddle.multinomial(probs, 1)
            out = paddle.concat((out, sample), axis=-1)

            pad_mask = paddle.full(shape=[mask.shape[0], 1], fill_value=1, dtype="bool")
            mask = paddle.concat((mask, pad_mask), axis=1)
            if (
                eos_token is not None
                and (
                    paddle.cumsum((out == eos_token).cast(paddle.int64), 1)[:, -1] >= 1
                ).all()
            ):
                break
            i_idx += 1
        out = out[:, t:]
        if num_dims == 1:
            out = out.squeeze(0)
        return out

    # forward for export
    def forward(self, inputs, targets=None):
        if not self.training:
            self.is_export = True
            encoded_feat = inputs
            batch_num = encoded_feat.shape[0]
            bos_tensor = paddle.full([batch_num, 1], self.bos_token, dtype=paddle.int64)
            if self.is_export:
                word_pred = self.generate_export(
                    bos_tensor,
                    self.max_seq_len,
                    eos_token=self.eos_token,
                    context=encoded_feat,
                    temperature=self.temperature,
                    filter_logits_fn=top_k,
                )
            else:
                word_pred = self.generate(
                    bos_tensor,
                    self.max_seq_len,
                    eos_token=self.eos_token,
                    context=encoded_feat,
                    temperature=self.temperature,
                    filter_logits_fn=top_k,
                )
            return word_pred

        encoded_feat, tgt_seq, mask = inputs
        kwargs = {"context": encoded_feat, "mask": mask.cast(paddle.bool)}
        x = tgt_seq
        xi = x[:, :-1]

        mask = kwargs.get("mask", None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs["mask"] = mask
        out = self.net(xi, **kwargs)

        return out
