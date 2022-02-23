# coding=utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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

import math

import numpy as np
import paddle

import paddle.nn as nn
import paddle.nn.functional as F

from ..model_utils import PretrainedModel, register_base_model
from ..nezha.modeling import ACT2FN

__all__ = [
    'T5Model',
    "T5PretrainedModel",
    'T5ForConditionalGeneration',
]


def finfo(dtype):
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)


class T5LayerNorm(nn.Layer):
    """
    Construct a layernorm module in the T5 style No bias and no subtraction of mean.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size],
            default_initializer=nn.initializer.Constant(1.0))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = paddle.pow(hidden_states.astype(paddle.float32), 2).mean(
            axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance +
                                                     self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == paddle.float16:
            hidden_states = hidden_states.astype(paddle.float16)
        return self.weight * hidden_states


class T5DenseReluDense(nn.Layer):
    """
    Construct a dense-relu-dense module.
    """

    def __init__(self, d_model, d_ff, dropout_rate):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias_attr=False)
        self.wo = nn.Linear(d_ff, d_model, bias_attr=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedGeluDense(nn.Layer):
    """
    Construct a dense-gated_gelu-dense module.
    """

    def __init__(self, d_model, d_ff, dropout_rate):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias_attr=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias_attr=False)
        self.wo = nn.Linear(d_ff, d_model, bias_attr=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Layer):
    def __init__(self, feed_forward_proj, d_model, d_ff, layer_norm_epsilon,
                 dropout_rate):
        super().__init__()
        if feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(d_model, d_ff, dropout_rate)
        elif feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(d_model, d_ff,
                                                        dropout_rate)
        else:
            raise ValueError(
                f"{feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )

        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Layer):
    def __init__(self,
                 is_decoder,
                 relative_attention_num_buckets,
                 d_model,
                 d_kv,
                 num_heads,
                 dropout_rate,
                 has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.d_model = d_model
        self.key_value_proj_dim = d_kv
        self.n_heads = num_heads
        self.dropout = dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias_attr=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  bidirectional=True,
                                  num_buckets=32,
                                  max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int64 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int64 values in the range [0, num_buckets)
        
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (
                relative_position > 0).astype(paddle.int64) * num_buckets
            relative_position = paddle.abs(relative_position)
        else:
            relative_position = -paddle.minimum(
                relative_position, paddle.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (paddle.log(
            relative_position.astype(paddle.get_default_dtype()) /
            max_exact) / math.log(max_distance / max_exact) * (
                num_buckets - max_exact)).astype(paddle.int64)
        relative_postion_if_large = paddle.minimum(
            relative_postion_if_large,
            paddle.full_like(relative_postion_if_large, num_buckets - 1), )

        relative_buckets += paddle.where(is_small, relative_position,
                                         relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = paddle.arange(query_length).unsqueeze(-1)
        memory_position = paddle.arange(key_length).unsqueeze(0)
        relative_position = (memory_position - context_position
                             )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets, )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.transpose(perm=[2, 0, 1]).unsqueeze(
            0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(self,
                hidden_states,
                mask=None,
                key_value_states=None,
                position_bias=None,
                cache=None,
                query_length=None,
                use_cache=False,
                output_attentions=False):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # cache[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if cache is not None:
            assert (
                len(cache) == 2
            ), f"cache should have 2 past states: keys and values. Got { len(cache)} past states"
            real_seq_length += (cache[0].shape[2]
                                if query_length is None else query_length)

        key_length = (real_seq_length if key_value_states is None else
                      key_value_states.shape[1])

        def shape(states):
            """projection"""
            return states.reshape(
                shape=[batch_size, -1, self.n_heads,
                       self.key_value_proj_dim]).transpose(perm=[0, 2, 1, 3])

        def unshape(states):
            """reshape"""
            return states.transpose(perm=[0, 2, 1, 3]).reshape(
                shape=[batch_size, -1, self.inner_dim])

        def project(hidden_states, proj_layer, key_value_states, cache):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif cache is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if cache is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = paddle.concat(
                        [cache, hidden_states], axis=2)
                else:
                    # cross-attn
                    hidden_states = cache
            return hidden_states

        # get query states
        query_states = shape(self.q(
            hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            cache[0] if cache is not None else None, )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            cache[1] if cache is not None else None, )

        # compute scores
        scores = paddle.matmul(query_states, key_states, transpose_y=True)

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = paddle.zeros(
                    shape=(1, self.n_heads, real_seq_length, key_length),
                    dtype=scores.dtype, )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if cache is not None:
                position_bias = position_bias[:, :, -hidden_states.shape[1]:, :]

            if mask is not None:
                position_bias = (
                    position_bias + mask
                )  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = F.softmax(
            scores.astype(paddle.float32), axis=-1).astype(
                scores.dtype)  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.
            training)  # (batch_size, n_heads, seq_length, key_length)

        attn_output = unshape(paddle.matmul(
            attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = ((key_states, value_states)
                                   if (self.is_decoder and use_cache) else None)
        outputs = (attn_output, ) + (present_key_value_state, ) + (
            position_bias, )

        if output_attentions:
            outputs = outputs + (attn_weights, )
        return outputs


class T5LayerSelfAttention(nn.Layer):
    def __init__(self,
                 is_decoder,
                 relative_attention_num_buckets,
                 d_model,
                 d_kv,
                 num_heads,
                 dropout_rate,
                 layer_norm_epsilon,
                 has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(
            is_decoder,
            relative_attention_num_buckets,
            d_model,
            d_kv,
            num_heads,
            dropout_rate,
            has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                hidden_states,
                attention_mask=None,
                position_bias=None,
                cache=None,
                use_cache=False,
                output_attentions=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            cache=cache,
            use_cache=use_cache,
            output_attentions=output_attentions, )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,
                   ) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Layer):
    def __init__(self, is_decoder, relative_attention_num_buckets, d_model,
                 d_kv, num_heads, dropout_rate, layer_norm_epsilon):
        super().__init__()
        self.EncDecAttention = T5Attention(
            is_decoder,
            relative_attention_num_buckets,
            d_model,
            d_kv,
            num_heads,
            dropout_rate,
            has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
            self,
            hidden_states,
            key_value_states,
            attention_mask=None,
            position_bias=None,
            cache=None,
            use_cache=False,
            query_length=None,
            output_attentions=False, ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            cache=cache,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions, )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,
                   ) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Layer):
    def __init__(self,
                 is_decoder,
                 relative_attention_num_buckets,
                 d_model,
                 d_kv,
                 num_heads,
                 dropout_rate,
                 layer_norm_epsilon,
                 feed_forward_proj,
                 d_ff,
                 has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.layer = nn.LayerList()
        self.layer.append(
            T5LayerSelfAttention(
                is_decoder,
                relative_attention_num_buckets,
                d_model,
                d_kv,
                num_heads,
                dropout_rate,
                layer_norm_epsilon,
                has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(
                T5LayerCrossAttention(
                    is_decoder, relative_attention_num_buckets, d_model, d_kv,
                    num_heads, dropout_rate, layer_norm_epsilon))

        self.layer.append(
            T5LayerFF(feed_forward_proj, d_model, d_ff, layer_norm_epsilon,
                      dropout_rate))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            cache=None,
            use_cache=False,
            output_attentions=False, ):

        if cache is not None:
            assert self.is_decoder, "Only decoder can use `caches`"
            expected_num_caches = 2 if encoder_hidden_states is None else 4

            if len(cache) != expected_num_caches:
                raise ValueError(
                    f"There should be {expected_num_caches} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_caches == 4 else ''}"
                    f"Got {len(cache)} past key / value states")

            self_attn_cache = cache[:2]
            cross_attn_cache = cache[2:]
        else:
            self_attn_cache, cross_attn_cache = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            cache=self_attn_cache,
            use_cache=use_cache,
            output_attentions=output_attentions, )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[
            2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == paddle.float16 and paddle.isinf(
                hidden_states).any():
            # TODO finfo
            clamp_value = finfo(hidden_states.dtype).max - 1000
            hidden_states = paddle.clip(
                hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                cache=cross_attn_cache,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions, )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if (hidden_states.dtype == paddle.float16 and
                    paddle.isinf(hidden_states).any()):
                clamp_value = finfo(hidden_states.dtype).max - 1000
                hidden_states = paddle.clip(
                    hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = (
                    present_key_value_state + cross_attention_outputs[1])

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == paddle.float16 and paddle.isinf(
                hidden_states).any():
            clamp_value = finfo(hidden_states.dtype).max - 1000
            hidden_states = paddle.clip(
                hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states, )

        if use_cache:
            outputs = outputs + (present_key_value_state, ) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5PretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained T5 models. It provides T5 related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    base_model_prefix = "t5"
    model_config_file = "model_config.json"

    pretrained_init_configuration = {
        "t5-small": {
            "tie_word_embeddings": True,
            "pad_token_id": 0,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "vocab_size": 32128,
            "d_model": 512,
            "d_kv": 64,
            "d_ff": 2048,
            "num_layers": 6,
            "num_decoder_layers": 6,
            "num_heads": 8,
            "relative_attention_num_buckets": 32,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-06,
            "initializer_factor": 1.0,
            "feed_forward_proj": "relu"
        },
        "t5-base": {
            "tie_word_embeddings": True,
            "pad_token_id": 0,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "vocab_size": 32128,
            "d_model": 768,
            "d_kv": 64,
            "d_ff": 3072,
            "num_layers": 12,
            "num_decoder_layers": 12,
            "num_heads": 12,
            "relative_attention_num_buckets": 32,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-06,
            "initializer_factor": 1.0,
            "feed_forward_proj": "relu"
        },
        "t5-large": {
            "tie_word_embeddings": True,
            "pad_token_id": 0,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "vocab_size": 32128,
            "d_model": 1024,
            "d_kv": 64,
            "d_ff": 4096,
            "num_layers": 24,
            "num_decoder_layers": 24,
            "num_heads": 16,
            "relative_attention_num_buckets": 32,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-06,
            "initializer_factor": 1.0,
            "feed_forward_proj": "relu"
        },
        "t5-v1_1-base": {
            "tie_word_embeddings": False,
            "pad_token_id": 0,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "vocab_size": 32128,
            "d_model": 768,
            "d_kv": 64,
            "d_ff": 2048,
            "num_layers": 12,
            "num_decoder_layers": 12,
            "num_heads": 12,
            "relative_attention_num_buckets": 32,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-06,
            "initializer_factor": 1.0,
            "feed_forward_proj": "gated-gelu",
        },
        "t5-v1_1-large": {
            "tie_word_embeddings": False,
            "pad_token_id": 0,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "vocab_size": 32128,
            "d_model": 1024,
            "d_kv": 64,
            "d_ff": 2816,
            "num_layers": 24,
            "num_decoder_layers": 24,
            "num_heads": 16,
            "relative_attention_num_buckets": 32,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-06,
            "initializer_factor": 1.0,
            "feed_forward_proj": "gated-gelu",
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "t5-small":
            "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-small/model_state.pdparams",
            "t5-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-base/model_state.pdparams",
            "t5-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-large/model_state.pdparams",
            "t5-v1_1-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-v1_1-base/model_state.pdparams",
            "t5-v1_1-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-v1_1-large/model_state.pdparams",
        }
    }

    @property
    def dummy_inputs(self):
        DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
        DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]
        input_ids = paddle.to_tensor(DUMMY_INPUTS, dtype=paddle.int64)
        input_mask = paddle.to_tensor(DUMMY_MASK, dtype=paddle.int64)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def init_weights(self):
        """
        Initializes and tie weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        """Initialize the weights"""
        factor = self.initializer_factor if hasattr(
            self, "initializer_factor") else self.t5.config[
                "initializer_factor"]  # Used for testing weights initialization
        d_model = self.d_model if hasattr(
            self, "d_model") else self.t5.config["d_model"]
        d_ff = self.d_ff if hasattr(self, "d_ff") else self.t5.config["d_ff"]
        n_heads = self.num_heads if hasattr(
            self, "num_heads") else self.t5.config["num_heads"]
        key_value_proj_dim = self.d_kv if hasattr(
            self, "d_kv") else self.t5.config["d_kv"]

        if isinstance(layer, T5LayerNorm):
            layer.weight.set_value(paddle.ones_like(layer.weight) * factor)
        elif isinstance(layer, T5Model):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            layer.shared.weight.set_value(
                paddle.normal(
                    mean=0.0, std=factor * 1.0,
                    shape=layer.shared.weight.shape))
        elif isinstance(layer, (T5ForConditionalGeneration, )):
            layer.t5.shared.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * 1.0,
                    shape=layer.t5.shared.weight.shape))

        elif isinstance(layer, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            layer.wi.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((d_model)**-0.5),
                    shape=layer.wi.weight.shape))

            if hasattr(layer.wi, "bias") and layer.wi.bias is not None:
                layer.wi.bias.set_value(paddle.zeros_like(layer.wi.bias))

            layer.wo.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((d_ff)**-0.5),
                    shape=layer.wo.weight.shape))

            if hasattr(layer.wo, "bias") and layer.wo.bias is not None:
                layer.wo.bias.set_value(paddle.zeros_like(layer.wo.bias))

        elif isinstance(layer, T5DenseGatedGeluDense):
            layer.wi_0.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((d_model)**-0.5),
                    shape=layer.wi_0.weight.shape))
            if hasattr(layer.wi_0, "bias") and layer.wi_0.bias is not None:
                layer.wi_0.bias.set_value(paddle.zeros_like(layer.wi_0.bias))

            layer.wi_1.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((d_model)**-0.5),
                    shape=layer.wi_1.weight.shape))
            if hasattr(layer.wi_1, "bias") and layer.wi_1.bias is not None:
                layer.wi_1.bias.set_value(paddle.zeros_like(layer.wi_1.bias))

            layer.wo.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((d_ff)**-0.5),
                    shape=layer.wo.weight.shape))

            if hasattr(layer.wo, "bias") and layer.wo.bias is not None:
                layer.wo.bias.set_value(paddle.zeros_like(layer.wo.bias))
        elif isinstance(layer, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136

            layer.q.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((d_model * key_value_proj_dim)**-0.5),
                    shape=layer.q.weight.shape))

            layer.k.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * (d_model**-0.5),
                    shape=layer.k.weight.shape))

            layer.v.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * (d_model**-0.5),
                    shape=layer.v.weight.shape))

            layer.o.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((n_heads * key_value_proj_dim)**-0.5),
                    shape=layer.o.weight.shape))

            if layer.has_relative_attention_bias:
                layer.relative_attention_bias.weight.set_value(
                    paddle.normal(
                        mean=0.0,
                        std=factor * ((d_model)**-0.5),
                        shape=layer.relative_attention_bias.weight.shape))

    def _shift_right(self, input_ids):
        bos_token_id = self.bos_token_id if hasattr(
            self, "bos_token_id") else self.t5.config["bos_token_id"]
        pad_token_id = self.pad_token_id if hasattr(
            self, "pad_token_id") else self.t5.config["pad_token_id"]

        assert (
            bos_token_id is not None
        ), "bos_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = paddle.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = bos_token_id

        assert (pad_token_id is not None), "pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = paddle.where(
            shifted_input_ids == -100,
            paddle.to_tensor(
                pad_token_id, dtype=shifted_input_ids.dtype),
            shifted_input_ids)

        assert paddle.all(
            shifted_input_ids >=
            0), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class T5Stack(nn.Layer):
    def __init__(self,
                 d_model,
                 num_layers,
                 layer_norm_epsilon,
                 dropout_rate,
                 relative_attention_num_buckets,
                 d_kv,
                 num_heads,
                 feed_forward_proj,
                 d_ff,
                 embed_tokens=None,
                 is_decoder=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.embed_tokens = embed_tokens
        self.block = nn.LayerList([
            T5Block(
                is_decoder,
                relative_attention_num_buckets,
                d_model,
                d_kv,
                num_heads,
                dropout_rate,
                layer_norm_epsilon,
                feed_forward_proj,
                d_ff,
                has_relative_attention_bias=bool(i == 0))
            for i in range(num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    @property
    def dtype(self):
        return self.embed_tokens.weight.dtype

    def forward(self,
                input_ids=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                cache=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False):
        assert input_ids is not None, "input_ids can not be None"
        input_shape = input_ids.shape
        input_ids = input_ids.reshape(shape=[-1, input_shape[-1]])

        inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (cache[0][0].shape[2] + seq_length
                           if cache is not None else seq_length)

        if use_cache is True:
            assert (
                self.is_decoder
            ), f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = paddle.ones(shape=[batch_size, mask_seq_length])
        if (self.is_decoder and encoder_attention_mask is None and
                encoder_hidden_states is not None):
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = paddle.ones(
                [batch_size, encoder_seq_length], dtype=paddle.int64)

        # initialize caches with `None` if past does not exist
        if cache is None:
            cache = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = paddle.ones(shape=encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and
                                      self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module,
                past_key_value) in enumerate(zip(self.block, cache)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                cache=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions, )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None, ) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state, )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3], )
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (
                        layer_outputs[5], )

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        return tuple(v
                     for v in [
                         hidden_states,
                         present_key_value_states,
                         all_hidden_states,
                         all_attentions,
                         all_cross_attentions,
                     ] if v is not None)

    def get_extended_attention_mask(self, attention_mask, input_shape):
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.ndim == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = paddle.arange(seq_length)
                causal_mask = paddle.tile(
                    seq_ids.unsqueeze(axis=[0, 1]),
                    [batch_size, seq_length, 1]) <= seq_ids.unsqueeze(
                        axis=[0, 2])
                # in case cache are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.astype(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[
                        1] - causal_mask.shape[1]
                    causal_mask = paddle.concat(
                        [
                            paddle.ones(
                                [batch_size, seq_length, prefix_seq_len],
                                dtype=causal_mask.dtype, ),
                            causal_mask,
                        ],
                        axis=-1, )

                extended_attention_mask = causal_mask.unsqueeze(
                    1) * attention_mask.unsqueeze([1, 2])
            else:
                extended_attention_mask = attention_mask.unsqueeze([1, 2])
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.astype(self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask):
        if encoder_attention_mask.ndim == 3:
            encoder_extended_attention_mask = encoder_attention_mask.unsqueeze(
                1)
        if encoder_attention_mask.ndim == 2:
            encoder_extended_attention_mask = encoder_attention_mask.unsqueeze(
                [1, 2])
        encoder_extended_attention_mask = encoder_extended_attention_mask.astype(
            self.dtype)  # fp16 compatibility

        if self.dtype == paddle.float16:
            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask) * -1e4
        elif self.dtype == paddle.float32:
            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError(
                f"{self.dtype} not recognized. `dtype` should be set to either `paddle.float32` or `paddle.float16`"
            )

        return encoder_extended_attention_mask


@register_base_model
class T5Model(T5PretrainedModel):
    """
    The bare T5 Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        tie_word_embeddings (bool, optional):
            Whether to tie input and output embeddings. Defaults to `False`.
        pad_token_id (int, optional):
            The id of the `padding` token. Defaults to `0`.
        bos_token_id (int, optional):
            The id of the `bos` token. Defaults to `0`.
        eos_token_id (int, optional):
            The id of the `eos` token. Defaults to `1`.
        initializer_factor (float, optional):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing). Defaults to `1.0`.
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in `T5Model`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `T5Model`. Defaults to `32128`.
        d_model (int, optional):
            Dimensionality of the embedding layer, encoder layer. Defaults to `768`.
        d_kv (int, optional):
            Size of the key, query, value projections per attention head. Defaults to `64`.
        d_ff (int, optional):
            Dimensionality of the feed_forward layer in the residual attention block. Defaults to `3072`.
        num_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        num_decoder_layers (int, optional):
            Number of hidden layers in the Transformer decoder. Defaults to `12`.
        num_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder and decoder.
            Defaults to `12`.
        relative_attention_num_buckets (int, optional):
            The number of buckets to use for each attention layer. Defaults to `32`.
        dropout_rate (float, optional):
            The dropout ratio for all layers. Defaults to `0.1`.
        layer_norm_eps (float, optional):
            The epsilon used by the layer normalization layers. Defaults to `1e-6`.
        feed_forward_proj (str, optional):
            The non-linear activation function (function or string) in the feed forward layer in the residual attention block. If string, `"relu"`, `"gated-gelu"` are supported. Defaults to `"relu"`.
        feed_forward_proj (str, optional):
            The non-linear activation function (function or string) in the feed forward layer in the residual attention block. If string, `"relu"`, `"gated-gelu"` are supported. Defaults to `"relu"`.

    """

    def __init__(self,
                 tie_word_embeddings=True,
                 pad_token_id=0,
                 bos_token_id=0,
                 eos_token_id=1,
                 initializer_factor=1.0,
                 vocab_size=32128,
                 d_model=768,
                 d_kv=64,
                 d_ff=3072,
                 num_layers=12,
                 num_decoder_layers=12,
                 num_heads=12,
                 relative_attention_num_buckets=32,
                 dropout_rate=0.1,
                 layer_norm_epsilon=1e-06,
                 feed_forward_proj="relu"):
        super().__init__()
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_kv = d_kv
        self.d_model = d_model
        self.initializer_factor = initializer_factor

        self.shared = nn.Embedding(vocab_size, d_model)
        self.encoder = T5Stack(
            d_model,
            num_layers,
            layer_norm_epsilon,
            dropout_rate,
            relative_attention_num_buckets,
            d_kv,
            num_heads,
            feed_forward_proj,
            d_ff,
            self.shared,
            is_decoder=False)
        self.decoder = T5Stack(
            d_model,
            num_decoder_layers,
            layer_norm_epsilon,
            dropout_rate,
            relative_attention_num_buckets,
            d_kv,
            num_heads,
            feed_forward_proj,
            d_ff,
            self.shared,
            is_decoder=True)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_output=None,
                cache=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False):
        r"""
        The T5Model forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on 
                to some unwanted positions, usually the paddings or the subsequent positions.
                Its data type can be int, float.
                When the data type is int, the `masked` tokens have `0` values and the others 
                have `1` values.
                When the data type is float, the `masked` tokens have `0.0` values and the 
                others have `1.0` values.
                It is a tensor with shape broadcasted to [batch_size, num_attention_heads, sequence_length, sequence_length].
                Defaults to `None`, which means nothing needed to be prevented attention to.
            decoder_input_ids (Tensor, optional):
                Indices of decoder input sequence tokens in the vocabulary.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means no `decoder_input_ids` is provided, the model will create the tensor
                by shifting the `input_ids` to the right.
            decoder_attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention to some unwanted positions in `decoder_input_ids`.
                Its data type and shape is the same as `attention_mask`. Defaults to `None`.
            encoder_output (tuple, optional):
                The output of the encoder, a tuple consists `last_hidden_state`, `hidden_states`(optional), `attentions`(optional).
                The data type of `last_hidden_state` is float32 and its shape is [batch_size, sequence_length, hidden_size].
                `hidden_states` is hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].
                `attentions` is attentions of all layers of in the Transformer encoder. The length of `attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is [batch_size, num_attention_heads, sequence_length, sequence_length].
            cache (Tuple[Tuple[Tensor]], optional):
                Contains pre-computed hidden-states (key and values in the attention blocks) 
                as computed by the model. Can be used to speed up sequential decoding. 
                The `input_ids` which have their past given to this model should not be 
                passed as input ids as they have already been computed.
                Defaults to `None`.
            use_cache (bool, optional):
                Whether or not to use cache. If set to `True`, `past_buckets_states` states are returned 
                and can be used to speed up decoding. 
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether or not to return the attentions tensors of all attention layers.
                Defaults to `False`.
            output_hidden_states (bool, optional):
                Whether or not to return the output of all hidden layers.
                Defaults to `False`.

        Returns:
            tuple: Returns tuple (`last_hidden_state`, `cache`, `decoder_hidden_states`, `decoder_attentions`,
            `cross_attentions`, `encoder_last_hidden_state`, `encoder_hidden_states`, `encoder_attentions`)

            With the fields:

            - `last_hidden_state` (Tensor):
                Sequence of hidden-states at the last layer of the decoder of the model.
                It's data type should be float32 and 
                its shape is [batch_size, sequence_length, hidden_size].

            - `cache` (List[tuple(Tensor, Tensor)], optional):
                returned when `use_cache=True` is passed.
                List of `tuple(Tensor, Tensor)` of length `config["num_layers"]`, 
                with the first element being the previous `buckets` of shape 
                `[batch_size, num_heads, num_hashes, sequence_length]` and the second 
                being the previous `hidden_states` of shape `[batch_size, sequence_length, hidden_size]`.

            - `decoder_hidden_states` (tuple(Tensor), optional)
                returned when ``output_hidden_states=True`` is passed.
                Tuple of `Tensor` (one for the output of the embeddings + one for the output of decoder each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            - `decoder_attentions` (tuple(Tensor), optional):
                returned when `output_attentions=True` is passed.
                tuple of `Tensor` (one for each layer) of shape. Each Tensor has a data 
                type of float32 and its shape is [batch_size, num_heads, sequence_length, sequence_length].

            - `cross_attentions` (tuple(Tensor), optional):
                returned when `output_attentions=True` is passed.
                tuple of `Tensor` (one for each layer) of shape. Each Tensor has a data 
                type of float32 and its shape is [batch_size, num_heads, sequence_length, sequence_length].

            - `encoder_last_hidden_state` (Tensor):
                Sequence of hidden-states at the last layer of the encoder of the model.
                It's data type should be float32 and 
                its shape is [batch_size, sequence_length, hidden_size].

            - `encoder_hidden_states` (tuple(Tensor), optional):
                returned when `output_hidden_states=True` is passed.
                tuple of `Tensor` (one for the output of the embeddings + one for the 
                output of encoder each layer). Each Tensor has a data type of float32 
                and its shape is [batch_size, sequence_length, hidden_size].

            - `encoder_attentions` (tuple(Tensor), optional):
                returned when `output_attentions=True` is passed.
                tuple of `Tensor` (one for each layer) of shape. Each Tensor has a data 
                type of float32 and its shape is [batch_size, num_heads, sequence_length, sequence_length].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import T5Model, T5Tokenizer

                tokenizer = T5Tokenizer.from_pretrained('t5-base')
                model = T5Model.from_pretrained('t5-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                outputs = model(**inputs)
                last_hidden_state = outputs[0]

        """

        # Encode if needed (training, first prediction pass)
        if encoder_output is None:
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states)

        hidden_states = encoder_output[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            cache=cache,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)

        return decoder_outputs + encoder_output


class T5ForConditionalGeneration(T5PretrainedModel):
    """
    The T5 Model transformer with a language modeling head on top.

    Args:
        t5 (:class:`T5Model`):
            An instance of :class:`T5Model`.

    """

    def __init__(self, t5):
        super().__init__()
        self.t5 = t5
        if not self.t5.config["tie_word_embeddings"]:
            self.lm_head = nn.Linear(
                self.t5.config["d_model"],
                self.t5.config["vocab_size"],
                bias_attr=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.t5.shared

    def set_input_embeddings(self, new_embeddings):
        self.t5.shared = new_embeddings
        self.t5.encoder.set_input_embeddings(new_embeddings)
        self.t5.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        if not self.t5.config["tie_word_embeddings"]:
            return self.t5.shared
        return self.lm_head

    def get_encoder(self):
        return self.t5.encoder

    def get_decoder(self):
        return self.t5.decoder

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_output=None,
                cache=None,
                labels=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False):
        r"""

        Args:
            input_ids (Tensor, optional):
                See :class:`T5Model`.
            attention_mask (Tensor, optional):
                See :class:`T5Model`.
            decoder_input_ids (Tensor, optional):
                See :class:`T5Model`.
            decoder_attention_mask (Tensor, optional):
                See :class:`T5Model`.
            encoder_output (tuple(Tensor), optional):
                See :class:`T5Model`.
            cache (List[tuple(Tensor, Tensor)], optional):
                See :class:`T5Model`.
            labels (Tensor, optional):
                Labels for language modeling. Note that the labels **are shifted** 
                inside the model, i.e. you can set `labels = input_ids` Indices are 
                selected in `[-100, 0, ..., vocab_size]` All labels set to `-100` are 
                ignored (masked), the loss is only computed for labels in `[0, ..., vocab_size]`.
                Shape is [batch_size, sequence_length] and dtype is int64.
            use_cache (bool, optional):
                See :class:`T5Model`.
            output_attentions (bool, optional):
                See :class:`T5Model`.
            output_hidden_states (bool, optional):
                See :class:`T5Model`.

        Returns:
            tuple: Returns tuple (`loss`, `logits`, `cache`, `decoder_hidden_states`, `decoder_attentions`,
            `cross_attentions`, `encoder_last_hidden_state`, `encoder_hidden_states`, `encoder_attentions`)

            With the fields:

            - `loss` (Tensor):
                returned when `labels` is provided.
                Language modeling loss. It's data type should be float32 and its shape is [1,].

            - `logits` (Tensor):
                Prediction scores of the language modeling head 
                (scores for each vocabulary token before SoftMax).
                It's data type should be float32 and its shape is 
                [batch_size, sequence_length, vocab_size].

            - `cache` (List[tuple(Tensor, Tensor)], optional):
                See :class:`T5Model`.

            - `decoder_hidden_states` (tuple(Tensor), optional)
                See :class:`T5Model`.

            - `decoder_attentions` (tuple(Tensor), optional):
                See :class:`T5Model`.

            - `cross_attentions` (tuple(Tensor), optional):
                See :class:`T5Model`.

            - `encoder_last_hidden_state` (Tensor):
                See :class:`T5Model`.

            - `encoder_hidden_states` (tuple(Tensor), optional):
                See :class:`T5Model`.

            - `encoder_attentions` (tuple(Tensor), optional):
                See :class:`T5Model`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import T5ForConditionalGeneration, T5Tokenizer

                tokenizer = T5Tokenizer.from_pretrained('t5-base')
                model = T5ForConditionalGeneration.from_pretrained('t5-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs, labels=inputs["input_ids"])

                loss = output[0]
                logits = output[1]

        """

        # Encode if needed (training, first prediction pass)
        if encoder_output is None:
            # Convert encoder inputs in embeddings if needed
            encoder_output = self.t5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states)

        hidden_states = encoder_output[0]

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if cache is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

        # Decode
        decoder_outputs = self.t5.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            cache=cache,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)

        sequence_output = decoder_outputs[0]

        if self.t5.config["tie_word_embeddings"]:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.t5.config["d_model"]
                                                 **-0.5)
            lm_logits = paddle.matmul(
                sequence_output, self.t5.shared.weight, transpose_y=True)
        else:
            lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                lm_logits.reshape(shape=[-1, lm_logits.shape[-1]]),
                labels.flatten())

        output = (lm_logits, ) + decoder_outputs[1:] + encoder_output
        return ((loss, ) + output) if loss is not None else output

    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no "
                             "`input_ids` are provided.")
        if encoder_output is not None:
            if isinstance(encoder_output, tuple):
                encoder_output = encoder_output[0]
            batch_size = encoder_output.shape[0]
        return paddle.ones([batch_size, 1], dtype="int64") * bos_token_id

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      cache=None,
                                      attention_mask=None,
                                      use_cache=None,
                                      encoder_output=None,
                                      **kwargs):

        # cut decoder_input_ids if past is used
        if cache is not None:
            input_ids = input_ids[:, -1:]
        return {
            "decoder_input_ids": input_ids,
            "cache": cache,
            "encoder_output": encoder_output,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    @staticmethod
    def expand_inputs_for_generation(input_ids,
                                     expand_size,
                                     attention_mask=None,
                                     **model_kwargs):
        index = paddle.tile(
            paddle.arange(input_ids.shape[0]).unsqueeze(-1),
            [1, expand_size]).reshape([-1])

        input_ids = paddle.index_select(input_ids, index)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = paddle.index_select(attention_mask,
                                                                 index)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.index_select(token_type_ids,
                                                                 index)

        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.index_select(position_ids,
                                                               index)

        if "seq_len" in model_kwargs:
            seq_len = model_kwargs["seq_len"]
            model_kwargs["seq_len"] = paddle.index_select(seq_len, index)

        if "encoder_output" in model_kwargs:
            encoder_output = model_kwargs["encoder_output"]
            if isinstance(encoder_output, tuple):
                model_kwargs["encoder_output"] = (paddle.index_select(
                    encoder_output[0], index), ) + encoder_output[1:]
            else:
                model_kwargs["encoder_output"] = paddle.index_select(
                    encoder_output, index)
        return input_ids, model_kwargs

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id,
                                              eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id))
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype("int64")
            return attention_mask
        else:
            attention_mask = paddle.ones_like(input_ids)
        return attention_mask

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            try:
                return getattr(getattr(self, self.base_model_prefix), name)
            except AttributeError:
                try:
                    return getattr(self, self.base_model_prefix).config[name]
                except KeyError:
                    raise e
