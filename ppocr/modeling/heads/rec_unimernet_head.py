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
https://github.com/opendatalab/UniMERNet/blob/main/unimernet/models/unimernet/configuration_unimernet_decoder.py
"""

import copy
import math
import re
import numpy as np
import inspect
import warnings
from collections import OrderedDict
from typing import Optional, Tuple, Union, List, Dict, Any
from dataclasses import dataclass, fields, is_dataclass

import paddle
import paddle.nn as nn
from paddle import Tensor
import paddle.nn.functional as F
from paddle.nn import CrossEntropyLoss
from paddle.nn.initializer import (
    TruncatedNormal,
    Constant,
    Normal,
    KaimingUniform,
    XavierUniform,
    XavierNormal,
)

zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)
kaiming_normal_ = KaimingUniform(nonlinearity="relu")
trunc_normal_ = TruncatedNormal(std=0.02)
xavier_uniform_ = XavierUniform()
xavier_normal_ = XavierNormal()


class ModelOutput(OrderedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        class_fields = fields(self)

        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(
                f"{self.__class__.__name__} should not have more than one required field."
            )

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:]
        )
        if other_fields_are_none:
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            self[class_fields[0].name] = first_field
                        else:
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
        )

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self):
        return tuple(self[k] for k in self.keys())


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    last_hidden_state = None
    past_key_values = None
    hidden_states = None
    attentions = None
    cross_attentions = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@dataclass
class Seq2SeqLMOutput(ModelOutput):
    loss = None
    logits = None
    past_key_values = None
    decoder_hidden_states = None
    decoder_attentions = None
    cross_attentions = None
    encoder_last_hidden_state = None
    encoder_hidden_states = None
    encoder_attentions = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MBartConfig(object):

    model_type = "mbart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
    }

    def __init__(
        self,
        vocab_size=50265,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        output_hidden_states=False,
        use_return_dict=True,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        forced_eos_token_id=2,
        _attn_implementation="eager",
        hidden_size=1024,
        use_parallel=False,
        parallel_step=2,
        is_export=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = (
            scale_embedding  # scale factor will be sqrt(d_model) if True
        )
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.forced_eos_token_id = forced_eos_token_id
        self._attn_implementation = _attn_implementation
        self.use_parallel = use_parallel
        self.parallel_step = parallel_step
        self.is_export = is_export
        super().__init__()


@dataclass
class AttentionMaskConverter:
    """
    A utility class for converting attention masks used in transformer models.

    This class handles the conversion of attention masks based on whether the
    attention mechanism is causal (i.e., preventing information flow from future
    tokens to past tokens) and whether a sliding window approach is used.

    Attributes:
        is_causal (bool): Indicates if the attention mechanism is causal.
        sliding_window (Optional[int]): Specifies the size of the sliding window
                                        for local attention, if applicable.

    Args:
        is_causal (bool): Determines if the attention mask should enforce causality.
        sliding_window (Optional[int], optional): The size of the sliding window
                                                  for local attention. Default is None.
    """

    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window=None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    @staticmethod
    def _make_causal_mask(
        input_ids_shape,
        dtype,
        past_key_values_length=0,
        sliding_window=None,
        is_export=False,
    ):
        bsz, tgt_len = input_ids_shape
        if is_export:
            mask = paddle.full(
                (tgt_len, tgt_len), paddle.finfo(dtype).min, dtype="float64"
            )
        else:
            mask = paddle.full((tgt_len, tgt_len), paddle.finfo(dtype).min)
        mask_cond = paddle.arange(mask.shape[-1])
        mask = mask.masked_fill_(
            mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1]), 0
        )
        return mask[None, None, :, :].expand(
            [bsz, 1, tgt_len, tgt_len + past_key_values_length]
        )

    def to_4d_export(
        self,
        attention_mask_2d,
        query_length,
        dtype,
        key_value_length,
        is_export=False,
    ):
        input_shape = (attention_mask_2d.shape[0], query_length)
        expanded_attn_mask = self._expand_mask(
            attention_mask_2d, dtype, tgt_len=input_shape[-1]
        )
        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask

    def to_4d(
        self,
        attention_mask_2d,
        query_length,
        dtype,
        key_value_length,
        is_export=False,
    ):

        input_shape = (attention_mask_2d.shape[0], query_length)
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length

            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
                is_export=is_export,
            )
        elif self.sliding_window is not None:
            raise NotImplementedError(
                "Sliding window is currently only implemented for causal masking"
            )

        expanded_attn_mask = self._expand_mask(
            attention_mask_2d, dtype, tgt_len=input_shape[-1]
        )

        if causal_4d_mask is not None:
            if is_export:
                expanded_attn_mask = causal_4d_mask
                return expanded_attn_mask
            else:
                expanded_attn_mask = causal_4d_mask.masked_fill_(
                    expanded_attn_mask.cast(paddle.bool), paddle.finfo(dtype).min
                )

        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask

    def _expand_mask(self, mask, dtype, tgt_len=None):
        bsz, src_len = mask.shape
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = (
            mask[:, None, None, :].expand([bsz, 1, tgt_len, src_len]).cast(dtype)
        )
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill_(
            inverted_mask.cast(paddle.bool), paddle.finfo(dtype).min
        )


def _prepare_4d_attention_mask(mask, dtype, tgt_len=None):
    return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _prepare_4d_causal_attention_mask_export(
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
    sliding_window=None,
    is_export=False,
):

    attn_mask_converter = AttentionMaskConverter(
        is_causal=True, sliding_window=sliding_window
    )
    key_value_length = input_shape[-1] + past_key_values_length

    shape = attention_mask.shape
    len_shape = len(shape)

    attention_mask = attn_mask_converter.to_4d_export(
        attention_mask,
        input_shape[-1],
        key_value_length=key_value_length,
        dtype=inputs_embeds.dtype,
        is_export=is_export,
    )
    return attention_mask


def _prepare_4d_causal_attention_mask(
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
    sliding_window=None,
    is_export=False,
):

    attn_mask_converter = AttentionMaskConverter(
        is_causal=True, sliding_window=sliding_window
    )
    key_value_length = input_shape[-1] + past_key_values_length

    shape = attention_mask.shape
    len_shape = len(shape)
    if (attention_mask is not None) and (len_shape == 2):
        attention_mask = attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            key_value_length=key_value_length,
            dtype=inputs_embeds.dtype,
            is_export=is_export,
        )

        return attention_mask
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill_(
                inverted_mask.to(paddle.bool), paddle.finfo(inputs_embeds.dtype).min
            )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0],
            input_shape[-1],
            key_value_length,
            dtype=inputs_embeds.dtype,
        )

    return attention_mask


class MBartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings, embedding_dim):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        positions = paddle.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=paddle.int64
        ).expand([bsz, -1])
        return nn.Embedding.forward(self, positions + self.offset)


class MBartPreTrainedModel(nn.Layer):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MBartDecoderLayer", "MBartAttention"]
    _supports_flash_attn_2 = True

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _initialize_weights(self, module):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_hf_initialized", False):
            return
        self._init_weights(module)

    def post_init(self):
        self.apply(self._initialize_weights)

    def _init_weights(self, module):
        std = self.config.init_std
        normal_ = Normal(mean=0.0, std=std)
        if isinstance(module, nn.Linear):
            normal_(module.weight)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight)
            if module._padding_idx is not None:
                zeros_(module.weight[module._padding_idx])

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = paddle.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]])
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class MBartAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.reshape([bsz, seq_len, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
    ):

        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = paddle.shape(hidden_states)
        query_states = self.q_proj(hidden_states) * self.scaling
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).reshape(proj_shape)
        key_states = key_states.reshape(proj_shape)
        value_states = value_states.reshape(proj_shape)

        src_len = key_states.shape[1]
        attn_weights = paddle.bmm(query_states, key_states.transpose([0, 2, 1]))

        if attention_mask is not None:
            attn_weights = (
                attn_weights.reshape([bsz, self.num_heads, tgt_len, src_len])
                + attention_mask
            )
            attn_weights = attn_weights.reshape(
                [bsz * self.num_heads, tgt_len, src_len]
            )

        attn_weights = nn.functional.softmax(attn_weights, axis=-1)
        if layer_head_mask is not None:
            if tuple(layer_head_mask.shape) != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of shape {(self.num_heads,)}, but is"
                    f" {layer_head_mask.shape}"
                )
            attn_weights = layer_head_mask.reshape(
                [1, -1, 1, 1]
            ) * attn_weights.reshape([bsz, self.num_heads, tgt_len, src_len])
            attn_weights = attn_weights.reshape(
                [bsz * self.num_heads, tgt_len, src_len]
            )

        if output_attentions:
            attn_weights_reshaped = attn_weights.reshape(
                [bsz, self.num_heads, tgt_len, src_len]
            )
            attn_weights = attn_weights_reshaped.reshape(
                [bsz * self.num_heads, tgt_len, src_len]
            )
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = paddle.bmm(attn_probs, value_states)

        attn_output = attn_output.reshape([bsz, self.num_heads, tgt_len, self.head_dim])
        attn_output = attn_output.transpose([0, 2, 1, 3])

        attn_output = attn_output.reshape([bsz, tgt_len, self.embed_dim])
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value


MBART_ATTENTION_CLASSES = {
    "eager": MBartAttention,
}


class MBartDecoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MBART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.is_export = config.is_export
        self.dropout = config.dropout
        self.activation_fn = F.gelu
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = MBART_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> paddle.Tensor:

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            hidden_states, cross_attn_weights, cross_attn_present_key_value = (
                self.encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states

            present_key_value = present_key_value + cross_attn_present_key_value

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if self.is_export:
            outputs += (present_key_value,)
        else:
            if use_cache:
                outputs += (present_key_value,)
        return outputs


class MBartForCausalLM(MBartPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        self.model = MBartDecoderWrapper(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            labels = labels
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.reshape([-1, self.config.vocab_size]), labels.reshape([-1])
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past


class myLayerNorm(nn.LayerNorm):
    """
    Custom implementation of Layer Normalization, with additional options.

    This class extends the standard LayerNorm to include optional features,
    such as drop block regularization, which might be used for improving
    model generalization.

    Args:
        num_channels (int): The number of features or channels in the input.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-5.
        affine (bool, optional): If True, this module has learnable affine parameters (gamma and beta). Default is True.
        drop_block (optional): Additional regularization technique that might be applied. Default is None.

    """

    def __init__(
        self,
        num_channels,
        eps=1e-5,
        affine=True,
        drop_block=None,
    ):
        super(nn.LayerNorm, self).__init__()
        self._epsilon = eps
        self.num_channels = num_channels
        if affine:
            self.weight = paddle.create_parameter([num_channels], dtype="float32")
            self.bias = paddle.create_parameter([num_channels], dtype="float32")
            ones_(self.weight)
            zeros_(self.bias)

    def forward(self, x):
        x = F.layer_norm(
            x,
            self.num_channels,
            weight=self.weight,
            bias=self.bias,
            epsilon=self._epsilon,
        )
        return x


class MBartDecoder(MBartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`MBartDecoderLayer`]

    Args:
        config
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.d_model, self.padding_idx
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.LayerList(
            [MBartDecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.layernorm_embedding = myLayerNorm(config.d_model, affine=True)
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        self.is_export = config.is_export

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.reshape([-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if self._use_flash_attention_2:
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                input_shape,
                inputs_embeds,
                past_key_values_length,
                is_export=self.is_export,
            )

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self._use_flash_attention_2:
                encoder_attention_mask = (
                    encoder_attention_mask if 0 in encoder_attention_mask else None
                )
            else:
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )
        next_decoder_cache = () if use_cache else None

        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                if attn_mask.shape[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {attn_mask.shape[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = paddle.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    (
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class MBartDecoderWrapper(MBartPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.decoder = MBartDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


def _in_projection(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    w_q: paddle.Tensor,
    w_k: paddle.Tensor,
    w_v: paddle.Tensor,
    b_q: Optional[paddle.Tensor] = None,
    b_k: Optional[paddle.Tensor] = None,
    b_v: Optional[paddle.Tensor] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:

    Eq, Ek, Ev = q.shape[-1], k.shape[-1], v.shape[-1]
    assert w_q.shape == (
        Eq,
        Eq,
    ), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (
        Eq,
        Ek,
    ), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (
        Eq,
        Ev,
    ), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (
        Eq,
    ), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (
        Eq,
    ), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (
        Eq,
    ), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q.T, b_q), linear(k, w_k.T, b_k), linear(v, w_v.T, b_v)


def _scaled_dot_product_attention(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    attn_mask: Optional[paddle.Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[paddle.Tensor, paddle.Tensor]:

    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    attn = paddle.bmm(q, k.transpose([0, 2, 1]))
    if attn_mask is not None:
        attn += attn_mask
    attn = F.softmax(attn, axis=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    output = paddle.bmm(attn, v)
    return output, attn


def linear(x, w, b, is_transpose):

    if b is not None:
        return paddle.matmul(x, w, transpose_y=is_transpose) + b
    else:
        return paddle.matmul(x, w, transpose_y=is_transpose)


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    is_export=False,
) -> List[Tensor]:

    E = paddle.shape(q)[-1]
    if k is v:
        if q is k:
            proj = linear(q, w, b, is_transpose=True)
            if is_export:
                B, D, L = paddle.shape(proj)
                proj = proj.reshape([B, D, 3, E])
                proj = (
                    proj.unsqueeze(0)
                    .transpose([3, 1, 2, 0, 4])
                    .squeeze(-2)
                    .contiguous()
                )
            else:
                proj = (
                    proj.unflatten(-1, (3, E))
                    .unsqueeze(0)
                    .transpose([3, 1, 2, 0, 4])
                    .squeeze(-2)
                    .contiguous()
                )
            return proj[0], proj[1], proj[2]
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def multi_head_attention_forward(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: paddle.Tensor,
    in_proj_bias: Optional[paddle.Tensor],
    bias_k: Optional[paddle.Tensor],
    bias_v: Optional[paddle.Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: paddle.Tensor,
    out_proj_bias: Optional[paddle.Tensor],
    training: bool = True,
    key_padding_mask: Optional[paddle.Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[paddle.Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[paddle.Tensor] = None,
    k_proj_weight: Optional[paddle.Tensor] = None,
    v_proj_weight: Optional[paddle.Tensor] = None,
    static_k: Optional[paddle.Tensor] = None,
    static_v: Optional[paddle.Tensor] = None,
    is_export=False,
):

    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    if isinstance(embed_dim, paddle.Tensor):
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    q, k, v = _in_projection_packed(
        query, key, value, in_proj_weight, in_proj_bias, is_export
    )

    if key_padding_mask is not None and key_padding_mask.dtype == paddle.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(paddle.bool)

    if bias_k is not None and bias_v is not None:  # False
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = paddle.concat([k, bias_k.repeat(1, bsz, 1)])
        v = paddle.concat([v, bias_v.repeat(1, bsz, 1)])
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.reshape([tgt_len, bsz * num_heads, head_dim]).transpose([1, 0, 2])
    if static_k is None:  # True
        k = k.reshape([k.shape[0], bsz * num_heads, head_dim]).transpose([1, 0, 2])
    else:
        assert (
            static_k.shape[0] == bsz * num_heads
        ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.shape[0]}"
        assert (
            static_k.shape[2] == head_dim
        ), f"expecting static_k.size(2) of {head_dim}, but got {static_k.shape[2]}"
        k = static_k
    if static_v is None:  # True
        v = v.reshape([v.shape[0], bsz * num_heads, head_dim]).transpose([1, 0, 2])
    else:
        assert (
            static_v.shape[0] == bsz * num_heads
        ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.shape[0]}"
        assert (
            static_v.shape[2] == head_dim
        ), f"expecting static_v.size(2) of {head_dim}, but got {static_v.shape[2]}"
        v = static_v

    src_len = k.shape[1]

    if not training:
        dropout_p = 0.0

    attn_output, attn_output_weights = _scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p
    )

    attn_output = attn_output.transpose([1, 0, 2]).reshape([tgt_len, bsz, embed_dim])
    attn_output = linear(
        attn_output, out_proj_weight, out_proj_bias, is_transpose=False
    )

    if need_weights:
        attn_output_weights = attn_output_weights.reshape(
            [bsz, num_heads, tgt_len, src_len]
        )
        return attn_output, attn_output_weights.sum(axis=1) / num_heads
    else:
        return attn_output, None


class MyMultiheadAttention(nn.Layer):
    """
    Custom implementation of a multi-head attention layer.

    Attributes:
        __constants__ (list): List of constant attributes.
        bias_k (Optional[paddle.Tensor]): Optional tensor for key bias.
        bias_v (Optional[paddle.Tensor]): Optional tensor for value bias.

    Args:
        embed_dim (int): Total dimension of the model. This is the size of the input feature vectors.
        num_heads (int): Number of parallel attention heads. The input dimension must be divisible by the number of heads.
        dropout (float, optional): Dropout probability on the attention weights. Default is 0.0.
        bias (bool, optional): If True, adds a learnable bias to the output. Default is True.
        add_bias_kv (bool, optional): If True, adds bias to the key and value sequences. Default is False.
        add_zero_attn (bool, optional): If True, adds a zero attention head. Default is False.
        kdim (int, optional): Total number of features for keys. If None, defaults to embed_dim.
        vdim (int, optional): Total number of features for values. If None, defaults to embed_dim.
        batch_first (bool, optional): If True, the input and output tensors are provided as (batch, seq, feature). Default is False.
        device (optional): The device on which the layer's parameters should be initialized. Default is None.
        dtype (optional): The data type for the parameters. Default is None.
        is_export (bool, optional): If True, the layer is set up for export, potentially changing behavior for compatibility. Default is False.
    """

    __constants__ = ["batch_first"]
    bias_k: Optional[paddle.Tensor]
    bias_v: Optional[paddle.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        is_export=False,
    ) -> None:
        super(MyMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.is_export = is_export
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            pass
        else:
            if dtype is None:
                dtype = paddle.float32
            self.in_proj_weight = paddle.create_parameter(
                (3 * embed_dim, embed_dim), dtype
            )
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None

        if bias:
            self.in_proj_bias = paddle.create_parameter((3 * embed_dim,), dtype)
            zeros_(self.in_proj_bias)
        else:
            self.in_proj_bias = None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)

        if add_bias_kv:
            pass
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):

        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            zeros_(self.in_proj_bias)
            zeros_(self.out_proj.bias)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        key_padding_mask: Optional[paddle.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:

        attn_output, attn_output_weights = multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_export=self.is_export,
        )

        return attn_output, attn_output_weights


class LogitsProcessorList(list):
    """
    A list of logits processors that can be applied sequentially.

    Methods:
        __call__(input_ids, scores, **kwargs): Apply all processors to the given inputs.
    """

    def __call__(self, input_ids, scores, **kwargs):
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


class ForcedEOSTokenLogitsProcessor(object):
    """
    A processor that forces the generation of an end-of-sequence (EOS) token
    at a specified position in the sequence.

    This is typically used in language generation tasks to ensure that the
    generated sequence ends properly when it reaches a certain length.

    Args:
        max_length (int): The maximum length of the sequence. Forces EOS when this length is reached.
        eos_token_id (Union[int, List[int]]): The ID(s) of the EOS token(s) to be forced in the sequence.
    """

    def __init__(self, max_length: int, eos_token_id: Union[int, List[int]]):
        self.max_length = max_length
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        scores_processed = scores
        if cur_len == self.max_length - 1:
            scores_processed = paddle.full_like(scores, -math.inf)
            scores_processed[:, self.eos_token_id] = 0
        return scores_processed


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):

    loss = None
    logits = None
    past_key_values = None
    hidden_states = None
    attentions = None
    cross_attentions = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@dataclass
class CausalLMOutputWithCrossAttentionsAndCounting(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    """

    logits = None
    counting = None
    past_key_values = None
    hidden_states = None
    attentions = None
    cross_attentions = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CustomMBartDecoder(MBartDecoder):
    """
    A custom MBartDecoder that includes additional processing layers.

    This class extends the MBartDecoder by adding a customizable neural network
    component called `counting_context_weight`, which applies a series of linear
    transformations followed by ReLU activations. This can be used to modify or
    enhance the decoder's behavior for specific tasks.

    Args:
        config: The configuration object containing model parameters.
    """

    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.d_model
        self.is_export = config.is_export
        self.counting_context_weight = nn.Sequential(
            nn.Linear(config.vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, config.d_model),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        count_pred=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.is_export = False if self.training else True
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.reshape([-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if self._use_flash_attention_2:
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            if self.is_export:
                attention_mask = _prepare_4d_causal_attention_mask_export(
                    attention_mask,
                    input_shape,
                    inputs_embeds,
                    past_key_values_length,
                    is_export=self.is_export,
                ).cast(paddle.float32)
            else:
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    input_shape,
                    inputs_embeds,
                    past_key_values_length,
                    is_export=self.is_export,
                )

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self._use_flash_attention_2:
                encoder_attention_mask = (
                    encoder_attention_mask if 0 in encoder_attention_mask else None
                )
            else:
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)

        hidden_states = inputs_embeds + positions

        # TODO: add counting context weight to hidden_states
        if count_pred is not None:
            count_context_weight = self.counting_context_weight(count_pred)
            hidden_states = hidden_states + 0.5 * count_context_weight.unsqueeze(1)

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {attn_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = paddle.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    (
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            if self.is_export:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            else:
                if use_cache:
                    next_decoder_cache += (
                        layer_outputs[3 if output_attentions else 1],
                    )

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if self.is_export:
            next_cache = next_decoder_cache
        else:
            next_cache = next_decoder_cache if use_cache else None
        if not self.is_export:
            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        next_cache,
                        all_hidden_states,
                        all_self_attns,
                        all_cross_attentions,
                    ]
                    if v is not None
                )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class SelfAttentionBlock(nn.Layer):
    """
    A self-attention block that implements multi-head self-attention
    followed by a feed-forward network, typically used in transformer architectures.

    Args:
        embed_size (int): The size of the embedding vector.
        num_heads (int): The number of attention heads.
        is_export (bool): Flag indicating whether to configure the layer for export.
    """

    def __init__(self, embed_size, num_heads, is_export):
        super(SelfAttentionBlock, self).__init__()
        self.self_attention = MyMultiheadAttention(
            embed_dim=embed_size, num_heads=num_heads, is_export=is_export
        )
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm(attn_output + x)
        return x


class SeqCountingDecoder(nn.Layer):
    """
    A custom sequence counting decoder that incorporates multi-head attention layers
    and feed-forward networks to process sequences, potentially for latex code counting .

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        num_heads (int): The number of attention heads. Defaults to 8.
        num_layers (int): The number of attention layers. Defaults to 4.
        is_export (bool): Flag indicating whether to configure the layer for export.
    """

    def __init__(
        self, in_features, out_features, num_heads=8, num_layers=4, is_export=False
    ):
        super(SeqCountingDecoder, self).__init__()

        self.attention_blocks = nn.LayerList(
            [
                SelfAttentionBlock(
                    embed_size=in_features, num_heads=num_heads, is_export=is_export
                )
                for i in range(num_layers)
            ]
        )
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1D(1)
        self.fc2 = nn.Linear(in_features // 2, out_features)

    def forward(self, x):
        for block in self.attention_blocks:
            x = block(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = x.transpose([0, 2, 1])
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.fc2(x)
        return x


class CustomMBartForCausalLM(MBartForCausalLM):
    """
    Custom MBart model for causal language modeling with a custom decoder.

    This class extends the MBartForCausalLM by replacing its decoder with a
    custom decoder, allowing for additional flexibility and features in the
    decoding process.

    Args:
        config: The configuration object containing model parameters.
        length_aware (bool): A flag to enable or configure length-aware mechanisms.
    """

    def __init__(self, config, length_aware=True):
        super().__init__(config)
        self.model.decoder = CustomMBartDecoder(config)
        self.counting_decoder = SeqCountingDecoder(
            config.d_model, config.vocab_size, is_export=config.is_export
        )
        self.length_aware = length_aware

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        count_gt=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.length_aware:
            count_pred = self.counting_decoder(encoder_hidden_states)
        else:
            count_pred = None

        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            count_pred=count_pred,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.lm_head(outputs[0])

        return CausalLMOutputWithCrossAttentionsAndCounting(
            logits=logits,
            counting=count_pred,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class UniMERNetHead(nn.Layer):
    """Implementation of UniMERNetHead decoder.

    Args:
         max_new_tokens (int): Maximum number of new tokens to generate.
         decoder_start_token_id (int): ID of the token that starts the decoding.
         temperature (float): Sampling temperature for generation.
         do_sample (bool): Whether to use sampling; if False, uses greedy decoding.
         top_p (float): Top-p (nucleus) sampling parameter.
         in_channels (int): Number of input channels/features.
         encoder_hidden_size (int): Hidden size of the encoder.
         decoder_hidden_size (int): Hidden size of the decoder.
         decoder_ffn_dim (int): Dimension of the decoder's feed-forward network.
         decoder_layers (int): Number of layers in the decoder.
         is_export (bool): Flag indicating if the model is being prepared for export.
         length_aware (bool): Flag to enable length-aware mechanisms.
    """

    def __init__(
        self,
        max_new_tokens=1536,
        decoder_start_token_id=0,
        temperature=0.2,
        do_sample=False,
        top_p=0.95,
        in_channels=1024,
        encoder_hidden_size=1024,
        decoder_hidden_size=1024,
        decoder_ffn_dim=4096,
        decoder_layers=8,
        is_export=False,
        length_aware=True,
    ):
        super().__init__()
        mbart_config_dict = {
            "activation_dropout": 0.0,
            "activation_function": "gelu",
            "add_cross_attention": True,
            "add_final_layer_norm": True,
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "classifier_dropout": 0.0,
            "d_model": decoder_hidden_size,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": decoder_ffn_dim,
            "decoder_layerdrop": 0.0,
            "decoder_layers": decoder_layers,
            "dropout": 0.1,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "encoder_layerdrop": 0.0,
            "encoder_layers": 12,
            "eos_token_id": 2,
            "forced_eos_token_id": 2,
            "init_std": 0.02,
            "is_decoder": True,
            "is_encoder_decoder": False,
            "output_hidden_states": False,
            "max_position_embeddings": max_new_tokens,
            "model_type": "mbart",
            "num_hidden_layers": 12,
            "pad_token_id": 1,
            "scale_embedding": True,
            "tie_word_embeddings": False,
            "transformers_version": "4.40.0",
            "use_cache": True,
            "use_return_dict": True,
            "vocab_size": 50000,
            "_attn_implementation": "eager",
            "hidden_size": decoder_hidden_size,
            "is_export": is_export,
        }

        self.max_new_tokens = max_new_tokens
        self.decoder_start_token_id = decoder_start_token_id
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.max_seq_len = max_new_tokens
        self.config_decoder = MBartConfig(**mbart_config_dict)
        self.encoder_hidden_size = encoder_hidden_size
        self.is_export = self.config_decoder.is_export
        self.decoder = CustomMBartForCausalLM(
            self.config_decoder, length_aware=length_aware
        )
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            self.enc_to_dec_proj = nn.Linear(
                self.encoder_hidden_size, self.config_decoder.hidden_size
            )
        generation_config = {
            "max_length": 1537,
            "forced_eos_token_id": 2,
        }
        self.eos_token_id = generation_config["forced_eos_token_id"]
        self.pad_token_id = self.config_decoder.pad_token_id
        self.logits_processor = LogitsProcessorList()
        self.logits_processor.append(
            ForcedEOSTokenLogitsProcessor(
                generation_config["max_length"],
                generation_config["forced_eos_token_id"],
            )
        )

    def _get_decoder_start_token_id(
        self, decoder_start_token_id=None, bos_token_id=None
    ) -> int:
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.generation_config.decoder_start_token_id
        )
        bos_token_id = (
            bos_token_id
            if bos_token_id is not None
            else self.generation_config.bos_token_id
        )
        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size,
        model_kwargs,
        decoder_start_token_id=None,
        bos_token_id=None,
    ):
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        decoder_start_token_id = self._get_decoder_start_token_id(
            decoder_start_token_id, bos_token_id
        )

        if isinstance(decoder_start_token_id, list):
            if len(decoder_start_token_id) != batch_size:
                raise ValueError(
                    f"`decoder_start_token_id` expected to have length {batch_size} but got {len(decoder_start_token_id)}"
                )
            decoder_input_ids_start = paddle.to_tensor(
                decoder_start_token_id,
                dtype=paddle.int64,
            )
            decoder_input_ids_start = decoder_input_ids_start.view(-1, 1)
        else:
            decoder_input_ids_start = (
                paddle.ones(
                    (batch_size, 1),
                    dtype=paddle.int64,
                )
                * decoder_start_token_id
            )

        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        elif (
            self.config.model_type == "vision-encoder-decoder"
            and "donut" in self.name_or_path.lower()
        ):
            pass
        elif self.config.model_type in ["whisper"]:
            pass
        elif (
            isinstance(decoder_start_token_id, int)
            and (decoder_input_ids[:, 0] != decoder_start_token_id).all().item()
        ) or (
            isinstance(decoder_start_token_id, paddle.Tensor)
            and (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all().item()
        ):
            decoder_input_ids = paddle.concat(
                [decoder_input_ids_start, decoder_input_ids], axis=-1
            )
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = paddle.cat(
                    (
                        paddle.ones_like(decoder_attention_mask)[:, :1],
                        decoder_attention_mask,
                    ),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    def prepare_inputs_for_generation_mbart(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        **kwargs,
    ):

        if attention_mask is None:
            attention_mask = paddle.ones(input_ids.shape)

        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        decoder_inputs = self.prepare_inputs_for_generation_mbart(
            input_ids, past_key_values=past_key_values
        )
        decoder_attention_mask = (
            decoder_inputs["attention_mask"]
            if "attention_mask" in decoder_inputs
            else None
        )
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def prepare_inputs_for_generation_export(
        self,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):

        input_dict = {
            "decoder_attention_mask": None,
            "use_cache": use_cache,
        }
        return input_dict

    def _extract_past_from_model_output(
        self, outputs: ModelOutput, standardize_cache_format: bool = False
    ):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states

        return past_key_values

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], axis=-1
            )

        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = paddle.concat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    axis=-1,
                )
        else:
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = paddle.concat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], 1)
                        ),
                    ],
                    axis=-1,
                )

        if (
            "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        return model_kwargs

    def stopping_criteria(self, input_ids):
        if self.is_export:
            return input_ids[:, -1] == paddle.to_tensor([self.eos_token_id])
        is_done = paddle.isin(input_ids[:, -1], paddle.to_tensor([self.eos_token_id]))
        return is_done

    def generate_single_iter(
        self,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        encoder_hidden_states = encoder_outputs[0]
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        kwargs_decoder = {}

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            inputs_embeds=None,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        return Seq2SeqLMOutput(
            loss=None,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    @paddle.no_grad()
    def generate(
        self,
        model_kwargs,
    ):
        """
        Generate sequences using the UniMERNetHead for inference tasks.

        Args:
            model_kwargs (dict): A dictionary of model configurations and inputs, which typically include:
                - encoder_outputs: Outputs from the encoder.
                - use_cache: Boolean flag to indicate if caching should be used.
                - output_attentions: Boolean flag for outputting attention scores.
                - output_hidden_states: Boolean flag for outputting hidden states.

        Returns:
            A tensor containing the generated sequences.
        """
        batch_size = model_kwargs["encoder_outputs"]["last_hidden_state"].shape[0]
        generation_config = {
            "decoder_start_token_id": 0,
            "bos_token_id": 0,
        }
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config["decoder_start_token_id"],
            bos_token_id=generation_config["bos_token_id"],
        )
        model_kwargs["key use_cache"] = True
        batch_size, cur_len = input_ids.shape

        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        model_kwargs["cache_position"] = paddle.arange(cur_len)
        pad_token_id = self.pad_token_id
        eos_token_id = [self.eos_token_id]
        eos_token = self.eos_token_id
        unfinished_sequences = paddle.ones(batch_size, dtype=paddle.int64)
        for idx in range(self.max_seq_len):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self.generate_single_iter(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            next_token_logits = outputs.logits[:, -1, :]

            next_tokens_scores = self.logits_processor(input_ids, next_token_logits)
            next_tokens = paddle.argmax(next_tokens_scores, axis=-1)
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )
            input_ids = paddle.concat([input_ids, next_tokens[:, None]], axis=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config_decoder.is_encoder_decoder,
            )
            unfinished_sequences = unfinished_sequences & ~self.stopping_criteria(
                input_ids
            ).cast(paddle.int64)

            if (
                eos_token is not None
                and (
                    paddle.cumsum((input_ids == eos_token).cast(paddle.int64), 1)[:, -1]
                    >= 1
                ).all()
            ):
                break

        return input_ids

    @paddle.no_grad()
    def generate_export(
        self,
        encoder_outputs,
        model_kwargs,
    ):
        batch_size = encoder_outputs["last_hidden_state"].shape[0]
        generation_config = {
            "decoder_start_token_id": 0,
            "bos_token_id": 0,
        }
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config["decoder_start_token_id"],
            bos_token_id=generation_config["bos_token_id"],
        )
        input_ids = input_ids.reshape([-1, 1])
        decoder_input_ids = input_ids
        model_kwargs["key use_cache"] = True
        batch_size, cur_len = input_ids.shape

        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        cache_position = paddle.arange(cur_len)
        pad_token_id = self.pad_token_id
        eos_token_id = [self.eos_token_id]
        eos_token = self.eos_token_id
        unfinished_sequences = paddle.ones([batch_size], dtype=paddle.int64)
        i_idx = paddle.full([], 0)
        past_key_values = []
        for i in range(8):
            init_arr = paddle.zeros([batch_size, 16, 0, 64])
            paddle.jit.api.set_dynamic_shape(init_arr, [-1, -1, -1, -1])
            cache = (init_arr, init_arr, init_arr, init_arr)
            past_key_values.append(cache)
        idx = 0
        while i_idx < paddle.to_tensor(self.max_seq_len):

            model_inputs = self.prepare_inputs_for_generation_export(
                past_key_values=past_key_values, **model_kwargs
            )
            decoder_attention_mask = model_inputs["decoder_attention_mask"]
            decoder_attention_mask = paddle.ones(input_ids.shape)
            paddle.jit.api.set_dynamic_shape(decoder_input_ids, [-1, -1])
            paddle.jit.api.set_dynamic_shape(decoder_attention_mask, [-1, -1])

            outputs = self.generate_single_iter(
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1, :]

            next_tokens_scores = self.logits_processor(input_ids, next_token_logits)
            next_tokens = paddle.argmax(next_tokens_scores, axis=-1)
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )
            input_ids = paddle.concat([input_ids, next_tokens.unsqueeze(1)], axis=-1)
            past_length = past_key_values[0][0].shape[2]
            decoder_input_ids = next_tokens.unsqueeze(1)
            past_key_values = outputs.past_key_values
            cache_position = cache_position[-1:] + 1
            unfinished_sequences = unfinished_sequences & ~self.stopping_criteria(
                input_ids
            ).cast(paddle.int64)
            if (
                eos_token is not None
                and (
                    paddle.cumsum((input_ids == eos_token).cast(paddle.int64), 1)[:, -1]
                    >= 1
                ).all()
            ):
                break

            i_idx += 1
        return input_ids

    def forwad_train(
        self,
        encoder_outputs,
        decoder_input_ids,
        decoder_attention_mask,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """
        Training for the UniMERNetHead.

        Args:
            encoder_outputs: Outputs from the encoder, used as input to the decoder.
            decoder_input_ids: Input IDs for the decoder.
            decoder_attention_mask: Attention mask for the decoder inputs.
            past_key_values: Cached key/values for faster decoding.
            decoder_inputs_embeds: Optional embeddings for the decoder inputs.
            labels: Target labels for calculating loss.
            use_cache: Whether to use cache during decoding.
            output_attentions: Whether to return attention scores.
            output_hidden_states: Whether to return hidden states.
            return_dict: Whether to return a dictionary of outputs.
            **kwargs: Additional keyword arguments.

        Returns:
            logits: The raw, unnormalized predictions from the model.
            count_pred: Optional prediction related to sequence length or other counts.
            masked_labels: The labels used during training, possibly masked.
        """
        labels = decoder_input_ids * 1
        labels = labels.masked_fill_(labels == self.pad_token_id, -100)
        input_decoder_input_ids = decoder_input_ids[:, :-1]
        input_decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_hidden_states = encoder_outputs[0]
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        kwargs_decoder = {}
        decoder_outputs = self.decoder(
            input_ids=input_decoder_input_ids,
            attention_mask=input_decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            inputs_embeds=None,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        logits = decoder_outputs.logits
        count_pred = decoder_outputs.counting
        return logits, count_pred, labels

    def forward(self, inputs, targets=None):
        """
        Forward pass for the UniMERNetHead, handling both training and inference.

        Args:
            inputs: The input data, which can vary based on training or inference.
            targets: The target labels, used only during training.

        Returns:
            During inference: Returns predicted latex code.
            During training: Returns logits, predicted counts, and masked labels.
        """
        self.is_export = False if self.training else True
        if not self.training:
            encoder_outputs = inputs
            if self.is_export:
                model_kwargs = {
                    "output_attentions": False,
                    "output_hidden_states": False,
                    "use_cache": True,
                }
                word_pred = self.generate_export(encoder_outputs, model_kwargs)
            else:
                model_kwargs = {
                    "output_attentions": False,
                    "output_hidden_states": False,
                    "use_cache": True,
                    "encoder_outputs": encoder_outputs,
                }
                word_pred = self.generate(model_kwargs)

            return word_pred

        encoder_outputs, tgt_seq, mask = inputs
        logits, count_pred, masked_labels = self.forwad_train(
            encoder_outputs, tgt_seq, mask
        )
        return logits, count_pred, masked_labels
