# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import inspect
import sys
from collections import namedtuple
from functools import reduce
from operator import mul

import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer
from .. import PretrainedModel, register_base_model
from ..nezha.modeling import ACT2FN

import paddle

from ...utils.log import logger

__all__ = [
    "ReformerModel",
    "ReformerPretrainedModel",
    "ReformerForSequenceClassification",
    "ReformerForQuestionAnswering",
    "ReformerModelWithLMHead",
    "ReformerForMaskedLM",
]
# Define named tuples for nn.Layers here
LSHSelfAttentionOutput = namedtuple(
    "LSHSelfAttentionOutput", ["hidden_states", "attention_probs", "buckets"])
LocalSelfAttentionOutput = namedtuple("LocalSelfAttentionOutput",
                                      ["hidden_states", "attention_probs"])
AttentionOutput = namedtuple("AttentionOutput",
                             ["hidden_states", "attention_probs", "buckets"])
ReformerOutput = namedtuple(
    "ReformerOutput",
    ["hidden_states", "attn_output", "attention_probs", "buckets"])
ReformerBackwardOutput = namedtuple(
    "ReformerBackwardOutput",
    ["attn_output", "hidden_states", "grad_attn_output", "grad_hidden_states"],
)
ReformerEncoderOutput = namedtuple(
    "ReformerEncoderOutput",
    ["hidden_states", "all_hidden_states", "all_attentions", "cache"], )


def _logsumexp(x, axis=-1, keepdim=False):
    # paddle.logsumexp don't support 5D Tensor
    if axis < 0:
        axis = x.ndim + axis
    if axis > 1:
        lse = paddle.logsumexp(x.flatten(0, 1), axis=axis - 1, keepdim=keepdim)
        orgshape = x.shape
        if keepdim:
            orgshape[axis] = 1
        else:
            orgshape = orgshape[:axis] + orgshape[axis + 1:]

        return lse.reshape(shape=orgshape)
    else:
        raise ValueError("axis must larger equal than 1.")


def _stable_argsort(vector, axis=-1):
    # this function scales the vector so that paddle.argsort is stable.
    # paddle.argsort is not stable on its own
    scale_offset = (paddle.arange(vector.shape[axis]).reshape(
        shape=[1, -1]).astype(vector.dtype))
    scale_offset = scale_offset.expand_as(vector)
    scaled_vector = vector.shape[axis] * vector + (scale_offset %
                                                   vector.shape[axis])
    return paddle.argsort(scaled_vector, axis=axis)


def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim,
                               *input_tensors):
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the
    dimension `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.
    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as
    directly applying `forward_fn` to `input_tensors`.
    """
    assert len(
        input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"
    tensor_shape = input_tensors[0].shape[chunk_dim]
    assert all(input_tensor.shape[chunk_dim] == tensor_shape
               for input_tensor in
               input_tensors), "All input tenors have to be of the same shape"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given")

    if chunk_size > 0:
        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}")

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(
            input_tensor.chunk(
                num_chunks, axis=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(
            forward_fn(*input_tensors_chunk)
            for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return paddle.concat(output_chunks, axis=chunk_dim)

    return forward_fn(*input_tensors)


def _get_least_common_mult_chunk_len(attn_layers, lsh_attn_chunk_length,
                                     local_attn_chunk_length):
    attn_types_set = set(attn_layers)
    if len(attn_types_set) == 1 and attn_layers[0] == "lsh":
        return lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_layers[0] == "local":
        return local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == set(["lsh", "local"]):
        return np.lcm(lsh_attn_chunk_length, local_attn_chunk_length)
    else:
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `attn_layers`: {attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only.")


def _get_min_chunk_len(attn_layers, lsh_attn_chunk_length,
                       local_attn_chunk_length):
    attn_types_set = set(attn_layers)
    if len(attn_types_set) == 1 and attn_layers[0] == "lsh":
        return lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_layers[0] == "local":
        return local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == set(["lsh", "local"]):
        return min(lsh_attn_chunk_length, local_attn_chunk_length)
    else:
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `attn_layers`: {attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only.")


class ReverseSort(PyLayer):
    """
    modified from https://github.com/huggingface/transformers/blob/fbf468b0573baddb1b9d1bb088a8b6d5c9303a7e/src/transformers/models/reformer/modeling_reformer.py#L982-L1011
    After chunked attention is applied which sorted clusters, original ordering has to be restored. Since customized
    backward function is used for Reformer, the gradients of the output vectors have to be explicitly sorted here.
    """

    @staticmethod
    def forward(ctx, out_vectors, logits, sorted_bucket_idx,
                undo_sorted_bucket_idx):
        # save sorted_bucket_idx for backprop
        with paddle.no_grad():
            ctx.sorted_bucket_idx = sorted_bucket_idx
            # undo sort to have correct order for next layer
            raw_shape = out_vectors.shape
            out_vectors = out_vectors.transpose(perm=[0, 1, 3, 2])
            expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(
                -2).expand_as(out_vectors)
            out_vectors = (paddle.index_sample(
                out_vectors.reshape([-1, raw_shape[2]]),
                expanded_undo_sort_indices.reshape([-1, raw_shape[2]]),
            ).reshape(out_vectors.shape).transpose(perm=[0, 1, 3, 2]))

            logits = paddle.index_sample(
                logits.reshape([-1, raw_shape[2]]),
                undo_sorted_bucket_idx.reshape([-1, raw_shape[2]]),
            ).reshape(raw_shape[:3])

        return out_vectors, logits

    @staticmethod
    def backward(ctx, grad_out_vectors, grad_logits):
        # get parameters saved in ctx
        sorted_bucket_idx = ctx.sorted_bucket_idx

        raw_shape = grad_out_vectors.shape
        grad_out_vectors = grad_out_vectors.transpose(perm=[0, 1, 3, 2])

        expanded_sorted_bucket_idx = sorted_bucket_idx.unsqueeze(-2).expand_as(
            grad_out_vectors)
        grad_out_vectors = (paddle.index_sample(
            grad_out_vectors.reshape([-1, raw_shape[2]]),
            expanded_sorted_bucket_idx.reshape([-1, raw_shape[2]]),
        ).reshape(grad_out_vectors.shape).transpose(perm=[0, 1, 3, 2]))

        grad_logits = paddle.index_sample(
            grad_logits.reshape([-1, raw_shape[2]]),
            sorted_bucket_idx.reshape([-1, raw_shape[2]]),
        ).reshape(raw_shape[:3])

        return grad_out_vectors, sorted_bucket_idx, None, None


class _ReversibleFunction(PyLayer):
    """
    modified from https://github.com/huggingface/transformers/blob/c016dbdbdaf79339ae6d275d4651dc9f380be055/src/transformers/models/reformer/modeling_reformer.py#L1568-L1677
    To prevent Paddle from performing the usual backpropagation, a customized backward function is implemented here.
    This way it is made sure that no memory expensive activations are saved during the forward pass.
    """

    @staticmethod
    def forward(
            ctx,
            hidden_states,
            layers,
            attention_mask,
            num_hashes,
            all_hidden_states,
            all_attentions,
            cache,
            use_cache,
            orig_sequence_length,
            output_hidden_states,
            output_attentions, ):
        all_buckets = ()

        # split duplicated tensor
        hidden_states, attn_output = paddle.chunk(
            hidden_states, chunks=2, axis=-1)

        for layer_id, layer in enumerate(layers):
            if output_hidden_states is True:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer(
                prev_attn_output=attn_output,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                num_hashes=num_hashes,
                cache=cache,
                use_cache=use_cache,
                orig_sequence_length=orig_sequence_length,
                output_attentions=output_attentions, )

            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            all_buckets = all_buckets + (layer_outputs.buckets, )

            if output_attentions:
                all_attentions.append(layer_outputs.attention_probs)

        # Add last layer
        if output_hidden_states is True:
            all_hidden_states.append(hidden_states)

        # attach params to ctx for backward
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach())
        ctx.layers = layers
        ctx.all_buckets = all_buckets
        ctx.attention_mask = attention_mask

        # Concatenate 2 RevNet outputs
        return paddle.concat([attn_output, hidden_states], axis=-1)

    @staticmethod
    def backward(ctx, grad_hidden_states):

        grad_attn_output, grad_hidden_states = paddle.chunk(
            grad_hidden_states, chunks=2, axis=-1)

        # retrieve params from ctx for backward
        (attn_output, hidden_states) = ctx.saved_tensor()

        # create tuple
        output = ReformerBackwardOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            grad_attn_output=grad_attn_output,
            grad_hidden_states=grad_hidden_states, )

        # free memory
        del grad_attn_output, grad_hidden_states, attn_output, hidden_states

        layers = ctx.layers
        all_buckets = ctx.all_buckets
        attention_mask = ctx.attention_mask

        for idx, layer in enumerate(layers[::-1]):
            # pop last buckets from stack
            buckets = all_buckets[-1]
            all_buckets = all_buckets[:-1]

            # backprop
            output = layer.backward_pass(
                next_attn_output=output.attn_output,
                hidden_states=output.hidden_states,
                grad_attn_output=output.grad_attn_output,
                grad_hidden_states=output.grad_hidden_states,
                attention_mask=attention_mask,
                buckets=buckets, )

        assert all_buckets == (
        ), "buckets have to be empty after backpropagation"
        grad_hidden_states = paddle.concat(
            [output.grad_attn_output, output.grad_hidden_states], axis=-1)

        # num of return vars has to match num of forward() args
        # return gradient for hidden_states arg and None for other args
        return grad_hidden_states, None


class AxialPositionEmbeddings(nn.Layer):
    """
    Constructs axial position embeddings. Useful for very long input sequences to save memory and time.
    """

    def __init__(
            self,
            axial_pos_shape=[128, 512],
            axial_pos_embds_dim=[256, 768],
            hidden_dropout_prob=0.2,
            attn_layers=[
                "local",
                "local",
                "lsh",
                "local",
                "local",
                "local",
                "lsh",
                "local",
                "local",
                "local",
                "lsh",
                "local",
            ],
            lsh_attn_chunk_length=256,
            local_attn_chunk_length=128,
            hidden_size=1024, ):
        super().__init__()
        self.axial_pos_shape = axial_pos_shape
        self.axial_pos_embds_dim = axial_pos_embds_dim
        self.dropout = hidden_dropout_prob

        self.least_common_mult_chunk_length = _get_least_common_mult_chunk_len(
            attn_layers=attn_layers,
            lsh_attn_chunk_length=lsh_attn_chunk_length,
            local_attn_chunk_length=local_attn_chunk_length, )
        self.weights = nn.ParameterList()

        if sum(self.axial_pos_embds_dim) != hidden_size:
            raise ValueError(
                f"Make sure that axial_pos_embds factors: {self.axial_pos_embds_dim} sum to "
                f"hidden_size: {hidden_size}")

        # create weights
        for axis, axial_pos_embd_dim in enumerate(self.axial_pos_embds_dim):
            # create expanded shapes
            ax_shape = [1] * len(self.axial_pos_shape)
            ax_shape[axis] = self.axial_pos_shape[axis]
            ax_shape = ax_shape + [axial_pos_embd_dim]

            self.weights.append(
                paddle.create_parameter(
                    shape=ax_shape,
                    dtype=paddle.get_default_dtype(),
                    default_initializer=nn.initializer.Constant(value=1.0), ))

    def forward(self, position_ids):
        # broadcast weights to correct shape
        batch_size = position_ids.shape[0]
        sequence_length = position_ids.shape[1]

        broadcasted_weights = [
            weight.expand(
                shape=[batch_size] + self.axial_pos_shape + weight.shape[-1:])
            for weight in self.weights
        ]

        if self.training is True:
            if reduce(mul, self.axial_pos_shape) != sequence_length:
                raise ValueError(
                    f"If training, make sure that axial_pos_shape factors: {self.axial_pos_shape} multiply to "
                    f"sequence length. Got prod({self.axial_pos_shape}) != sequence_length: {sequence_length}. "
                    f"You might want to consider padding your sequence length to {reduce(mul, self.axial_pos_shape)} "
                    "or changing axial_pos_shape.")

            if self.dropout > 0:
                weights = paddle.concat(broadcasted_weights, axis=-1)
                # permute weights so that 2D correctly drops dims 1 and 2
                transposed_weights = weights.transpose(perm=[0, 2, 1, 3])
                # drop entire matrix of last two dims (prev dims 1 and 2)
                dropped_transposed_weights = F.dropout2d(
                    transposed_weights, p=self.dropout, training=self.training)
                dropped_weights = dropped_transposed_weights.transpose(
                    perm=[0, 2, 1, 3])

                position_encodings = paddle.reshape(
                    dropped_weights, shape=[batch_size, sequence_length, -1])

            else:
                position_encodings = paddle.concat(
                    [
                        paddle.reshape(
                            weight, shape=[batch_size, sequence_length, -1])
                        for weight in broadcasted_weights
                    ],
                    axis=-1, )

        else:
            if reduce(mul, self.axial_pos_shape) < sequence_length:
                raise ValueError(
                    f"Make sure that axial_pos_shape factors: {self.axial_pos_shape} multiply at least to "
                    f"max(sequence_length, least_common_mult_chunk_length): max({sequence_length}, "
                    f"{self.least_common_mult_chunk_length}).")

            # compute how many columns are needed
            max_position_id = position_ids.max().item()
            required_pos_encodings_columns = -(-(max_position_id + 1) //
                                               self.axial_pos_shape[1])
            # cut to columns that are needed
            position_encodings = paddle.concat(
                [
                    weight[:, :required_pos_encodings_columns]
                    for weight in broadcasted_weights
                ],
                axis=-1, )
            position_encodings = paddle.reshape(
                position_encodings,
                shape=[batch_size, -1, position_encodings.shape[-1]])

            # select correct position encodings
            position_encodings = paddle.concat(
                [
                    paddle.index_select(
                        position_encodings[i], index=position_ids[i],
                        axis=0).unsqueeze(0) for i in range(batch_size)
                ],
                axis=0, )

        return position_encodings


class PositionEmbeddings(nn.Layer):
    """Constructs conventional position embeddings of shape `[max_pos_embeddings, hidden_size]`."""

    def __init__(self,
                 hidden_dropout_prob=0.2,
                 max_position_embeddings=65536,
                 hidden_size=1024):
        super().__init__()
        self.dropout = hidden_dropout_prob
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(self, position_ids):
        position_embeddings = self.embedding(position_ids)
        position_embeddings = F.dropout(
            position_embeddings, p=self.dropout, training=self.training)
        return position_embeddings


class ReformerEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(
            self,
            max_position_embeddings=65536,
            hidden_dropout_prob=0.2,
            axial_pos_embds=True,
            vocab_size=258,
            hidden_size=1024,
            axial_pos_shape=[128, 512],
            axial_pos_embds_dim=[256, 768],
            attn_layers=[
                "local",
                "local",
                "lsh",
                "local",
                "local",
                "local",
                "lsh",
                "local",
                "local",
                "local",
                "lsh",
                "local",
            ],
            lsh_attn_chunk_length=256,
            local_attn_chunk_length=128, ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.dropout = hidden_dropout_prob

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = (AxialPositionEmbeddings(
            axial_pos_shape=axial_pos_shape,
            axial_pos_embds_dim=axial_pos_embds_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_layers=attn_layers,
            lsh_attn_chunk_length=lsh_attn_chunk_length,
            local_attn_chunk_length=local_attn_chunk_length,
            hidden_size=hidden_size,
        ) if axial_pos_embds else PositionEmbeddings(
            hidden_dropout_prob=hidden_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size, ))

    def forward(self, input_ids, position_ids=None, start_idx_pos_encodings=0):
        seq_length = input_ids.shape[1]
        if position_ids is None:
            position_ids = paddle.arange(start_idx_pos_encodings,
                                         start_idx_pos_encodings + seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        inputs_embeds = self.word_embeddings(input_ids)

        if position_ids.shape[-1] > self.max_position_embeddings:
            raise ValueError(
                f"Sequence Length: {position_ids.shape[-1]} has to be larger equal than "
                f"max_position_embeddings {self.max_position_embeddings}.")

        # dropout
        embeddings = F.dropout(
            inputs_embeds, p=self.dropout, training=self.training)

        # add positional embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        return embeddings


class EfficientAttentionMixin:
    """
    A few utilities for nn.Layers in Reformer, to be used as a mixin.
    """

    def _look_adjacent(self, vectors, num_chunks_before, num_chunks_after):
        """
        Used to implement attention between consecutive chunks.

        Args:
            vectors: array of shape [batch_size, num_attention_heads, n_chunks, chunk_len, ...]
            num_chunks_before: chunks before current chunk to include in attention
            num_chunks_after: chunks after current chunk to include in attention

        Returns:
            tensor of shape [num_chunks, N * chunk_length, ...], where N = (1 + num_chunks_before + num_chunks_after).
        """
        if num_chunks_before == 0 and num_chunks_after == 0:
            return vectors

        slices = []
        for i in range(-num_chunks_before, num_chunks_after + 1):
            if i == 0:
                slices.append(vectors)
            else:
                slices.append(
                    paddle.concat(
                        [vectors[:, :, i:], vectors[:, :, :i]], axis=2))
        return paddle.concat(slices, axis=3)

    def _split_hidden_size_dim(self, x, num_attn_heads, attn_head_size):
        """
        splits hidden_size dim into attn_head_size and num_attn_heads
        """
        new_x_shape = x.shape[:-1] + [num_attn_heads, attn_head_size]
        x = x.reshape(shape=new_x_shape)
        return x.transpose(perm=[0, 2, 1, 3])

    def _merge_hidden_size_dims(self, x, num_attn_heads, attn_head_size):
        """
        merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        x = x.transpose(perm=[0, 2, 1, 3])
        return paddle.reshape(
            x, shape=[x.shape[0], -1, num_attn_heads * attn_head_size])

    def _split_seq_length_dim_to(self,
                                 vectors,
                                 dim_factor_1,
                                 dim_factor_2,
                                 num_attn_heads,
                                 attn_head_size=None):
        """
        splits sequence length dim of vectors into `dim_factor_1` and `dim_factor_2` dims
        """
        batch_size = vectors.shape[0]

        split_dim_shape = [
            batch_size, num_attn_heads, dim_factor_1, dim_factor_2
        ]

        if vectors.ndim == 4:
            return paddle.reshape(
                vectors, shape=split_dim_shape + [attn_head_size])
        elif vectors.ndim == 3:
            return paddle.reshape(vectors, shape=split_dim_shape)
        else:
            raise ValueError(
                f"Input vector rank should be one of [3, 4], but is: {vectors.ndim}"
            )


class LSHSelfAttention(nn.Layer, EfficientAttentionMixin):
    def __init__(
            self,
            lsh_attn_chunk_length=256,
            num_hashes=4,
            num_buckets=512,
            lsh_num_chunks_before=1,
            lsh_num_chunks_after=0,
            hash_seed=None,
            is_decoder=True,
            max_position_embeddings=65536,
            lsh_attention_probs_dropout_prob=0.1,
            num_attention_heads=8,
            attention_head_size=128,
            hidden_size=1024, ):
        super().__init__()

        self.chunk_length = lsh_attn_chunk_length
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.num_chunks_before = lsh_num_chunks_before
        self.num_chunks_after = lsh_num_chunks_after
        self.hash_seed = hash_seed
        self.is_decoder = is_decoder
        self.max_position_embeddings = max_position_embeddings

        self.dropout = lsh_attention_probs_dropout_prob

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = hidden_size

        # projection matrices
        self.query_key = nn.Linear(
            self.hidden_size, self.all_head_size, bias_attr=False)
        self.value = nn.Linear(
            self.hidden_size, self.all_head_size, bias_attr=False)

        # save mask value here. Need fp32 and fp16 mask values
        self.register_buffer("self_mask_value_float16", paddle.to_tensor(-1e3))
        self.register_buffer("self_mask_value_float32", paddle.to_tensor(-1e5))
        self.register_buffer("mask_value_float16", paddle.to_tensor(-1e4))
        self.register_buffer("mask_value_float32", paddle.to_tensor(-1e9))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            num_hashes=None,
            buckets=None,
            cache=None,
            use_cache=False,
            output_attentions=False,
            **kwargs, ):
        batch_size, sequence_length = hidden_states.shape[:2]

        # num hashes can optionally be overwritten by user
        num_hashes = num_hashes if num_hashes is not None else self.num_hashes

        do_cached_attention = use_cache and cache[1] is not None

        # check if cache shall be used and that hidden states are already cached
        if do_cached_attention:
            assert (
                sequence_length == 1
            ), f"At the moment, auto-regressive language generation is only possible one word at a time. Make sure that input sequence length {sequence_length} equals 1, when `cache` is passed."
            past_buckets = cache[0]
            past_states = cache[1]

            # get query vector
            query_vectors = self.query_key(hidden_states)
            query_vectors = self._split_hidden_size_dim(
                query_vectors, self.num_attention_heads,
                self.attention_head_size)

            if past_buckets is not None:
                (
                    key_value_hidden_states,
                    sorted_bucket_idx,
                    buckets, ) = self._get_relevant_hid_states_and_buckets(
                        query_vectors=query_vectors,
                        attention_mask=attention_mask,
                        num_hashes=num_hashes,
                        hidden_states=hidden_states,
                        past_states=past_states,
                        past_buckets=past_buckets, )

                query_key_vectors = self._query_per_attn_head(
                    key_value_hidden_states)
                value_vectors = self._value_per_attn_head(
                    key_value_hidden_states)

                # split key & value vectors by num hashes to apply
                # self attention on each separately
                query_key_vectors = self._split_seq_length_dim_to(
                    query_key_vectors,
                    num_hashes,
                    -1,
                    self.num_attention_heads,
                    self.attention_head_size, )
                value_vectors = self._split_seq_length_dim_to(
                    value_vectors,
                    num_hashes,
                    -1,
                    self.num_attention_heads,
                    self.attention_head_size, )
                # expand query vectors across hash dimension
                query_vectors = paddle.tile(
                    query_vectors.unsqueeze(2),
                    repeat_times=[1, 1, num_hashes, 1, 1])
            else:
                key_value_hidden_states = paddle.concat(
                    [past_states, hidden_states], axis=1)

                query_key_vectors = self.query_key(key_value_hidden_states)
                value_vectors = self.value(key_value_hidden_states)

        else:
            # project hidden_states to query_key and value
            query_vectors = None
            query_key_vectors = self.query_key(hidden_states)
            value_vectors = self.value(hidden_states)

        # if query key is not already split
        if not do_cached_attention or past_buckets is None:
            query_key_vectors = self._split_hidden_size_dim(
                query_key_vectors, self.num_attention_heads,
                self.attention_head_size)
            value_vectors = self._split_hidden_size_dim(
                value_vectors, self.num_attention_heads,
                self.attention_head_size)

        # cache buckets for next incremental decoding
        if (do_cached_attention and past_buckets is None and
                key_value_hidden_states.shape[1] >= self.chunk_length):
            buckets = self._hash_vectors(query_key_vectors, num_hashes,
                                         attention_mask)

        # free memory
        del hidden_states

        assert (
            query_key_vectors.shape[-1] == self.attention_head_size
        ), f"last dim of query_key_vectors is {query_key_vectors.shape[-1]} but should be {self.attention_head_size}."
        assert (
            value_vectors.shape[-1] == self.attention_head_size
        ), f"last dim of value_vectors is {value_vectors.shape[-1]} but should be {self.attention_head_size}."

        do_standard_self_attention = (sequence_length <= self.chunk_length) or (
            use_cache and cache[1] is not None)
        # LSH attention only makes sense if chunked attention should be performed
        if not do_standard_self_attention:
            # set `num_buckets` on the fly, recommended way to do it
            if self.num_buckets is None:
                self._set_num_buckets(sequence_length)

            # use cached buckets for backprop only
            if buckets is None:
                # hash query key vectors into buckets
                buckets = self._hash_vectors(query_key_vectors, num_hashes,
                                             attention_mask)
            else:
                # make sure buckets has correct shape for LSH attention
                buckets = buckets.reshape(shape=[
                    batch_size,
                    self.num_attention_heads,
                    num_hashes * sequence_length,
                ])

            assert (
                int(buckets.shape[-1]) == num_hashes * sequence_length
            ), f"last dim of buckets is {buckets.shape[-1]}, but should be {num_hashes * sequence_length}"

            (
                sorted_bucket_idx,
                undo_sorted_bucket_idx,
            ) = self._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(buckets)

            # make sure bucket idx is not longer then sequence length
            sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length

            # cluster query key value vectors according to hashed buckets
            query_key_vectors = self._gather_by_expansion(
                query_key_vectors, sorted_bucket_idx_per_hash, num_hashes)

            value_vectors = self._gather_by_expansion(
                value_vectors, sorted_bucket_idx_per_hash, num_hashes)
            query_key_vectors = self._split_seq_length_dim_to(
                query_key_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size, )
            value_vectors = self._split_seq_length_dim_to(
                value_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size, )

            if self.chunk_length is None:
                assert (
                    self.num_chunks_before == 0 and self.num_chunks_after == 0
                ), "If `chunk_length` is `None`, make sure `num_chunks_after` and `num_chunks_before` are set to 0."
        elif do_cached_attention and past_buckets is not None:
            # use max sequence length
            sorted_bucket_idx_per_hash = sorted_bucket_idx
        else:
            # get sequence length indices
            sorted_bucket_idx_per_hash = paddle.tile(
                paddle.arange(sequence_length),
                repeat_times=[batch_size, self.num_attention_heads, 1], )

        # scale key vectors
        key_vectors = self._len_and_dim_norm(query_key_vectors)

        # set query_vectors to query key vectors if LSH self attention
        query_vectors = (query_vectors
                         if query_vectors is not None else query_key_vectors)

        # free memory
        del query_key_vectors

        # get attention probs
        out_vectors, logits, attention_probs = self._attend(
            query_vectors=query_vectors,
            key_vectors=key_vectors,
            value_vectors=value_vectors,
            sorted_bucket_idx_per_hash=sorted_bucket_idx_per_hash,
            attention_mask=attention_mask,
            do_standard_self_attention=do_standard_self_attention,
            do_cached_attention=do_cached_attention, )

        # free memory
        del key_vectors, value_vectors

        # re-order out_vectors and logits
        if not do_standard_self_attention:
            # sort clusters back to correct ordering
            out_vectors, logits = ReverseSort.apply(
                out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx)

        if not do_standard_self_attention or (do_cached_attention and
                                              past_buckets is not None):
            # sum up all hash rounds
            if num_hashes > 1:
                out_vectors = self._split_seq_length_dim_to(
                    out_vectors,
                    num_hashes,
                    sequence_length,
                    self.num_attention_heads,
                    self.attention_head_size, )
                logits = self._split_seq_length_dim_to(
                    logits,
                    num_hashes,
                    sequence_length,
                    self.num_attention_heads,
                    self.attention_head_size, ).unsqueeze(-1)

                probs_vectors = paddle.exp(logits - _logsumexp(
                    logits, axis=2, keepdim=True))
                out_vectors = paddle.sum(out_vectors * probs_vectors, axis=2)
                # free memory
                del probs_vectors

            # free memory
            del logits

        assert out_vectors.shape == [
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        ], "out_vectors have be of shape `[batch_size, num_attention_heads, sequence_length, attention_head_size]`."

        out_vectors = self._merge_hidden_size_dims(
            out_vectors, self.num_attention_heads, self.attention_head_size)

        if output_attentions is False:
            attention_probs = ()

        if buckets is not None:
            buckets = buckets.reshape(
                shape=[batch_size, self.num_attention_heads, num_hashes, -1])

        return LSHSelfAttentionOutput(
            hidden_states=out_vectors,
            attention_probs=attention_probs,
            buckets=buckets)

    def _query_per_attn_head(self, hidden_states):
        per_head_query_key = self.query_key.weight.reshape(shape=[
            self.num_attention_heads, self.attention_head_size, self.hidden_size
        ]).transpose(perm=[0, 2, 1])
        # only relevant for inference and no bias => we can use einsum here
        query_key_vectors = paddle.einsum("balh,ahr->balr", hidden_states,
                                          per_head_query_key)
        return query_key_vectors

    def _value_per_attn_head(self, hidden_states):
        per_head_value = self.value.weight.reshape(shape=[
            self.num_attention_heads, self.attention_head_size, self.hidden_size
        ]).transpose(perm=[0, 2, 1])
        # only relevant for inference and no bias => we can use einsum here
        value_vectors = paddle.einsum("balh,ahr->balr", hidden_states,
                                      per_head_value)
        return value_vectors

    def _hash_vectors(self,
                      vectors,
                      num_hashes,
                      attention_mask,
                      increase_num_buckets=False):
        batch_size = vectors.shape[0]

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        if isinstance(self.num_buckets, int):
            assert (
                self.num_buckets % 2 == 0
            ), f"There should be an even number of buckets, but `self.num_buckets`: {self.num_buckets}"
            rotation_size = self.num_buckets
            num_buckets = self.num_buckets
        else:
            # Factorize the hash if self.num_buckets is a list or tuple
            rotation_size, num_buckets = 0, 1
            for bucket_factor in self.num_buckets:
                assert (
                    bucket_factor % 2 == 0
                ), f"The number of buckets should be even, but `num_bucket`: {bucket_factor}"
                rotation_size = rotation_size + bucket_factor
                num_buckets = num_buckets * bucket_factor

        # remove gradient
        vectors = vectors.detach()

        if self.hash_seed is not None:
            # for determinism
            paddle.seed(self.hash_seed)

        rotations_shape = [
            self.num_attention_heads,
            vectors.shape[-1],
            num_hashes,
            rotation_size // 2,
        ]

        # create a random self.attention_head_size x num_hashes x num_buckets/2
        random_rotations = paddle.randn(
            shape=rotations_shape, dtype=vectors.dtype)
        # Output dim: Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len x Num_Buckets/2
        rotated_vectors = paddle.einsum("bmtd,mdhr->bmhtr", vectors,
                                        random_rotations)

        if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
            rotated_vectors = paddle.concat(
                [rotated_vectors, -rotated_vectors], axis=-1)
            buckets = paddle.argmax(rotated_vectors, axis=-1)
        else:
            # Get the buckets for them and combine.
            buckets, cur_sum, cur_product = None, 0, 1
            for bucket_factor in self.num_buckets:
                # bmhtr
                rotated_vectors_factor = rotated_vectors[:, :, :, :, cur_sum:
                                                         cur_sum +
                                                         (bucket_factor // 2)]
                cur_sum = cur_sum + bucket_factor // 2
                rotated_vectors_factor = paddle.concat(
                    [rotated_vectors_factor, -rotated_vectors_factor], axis=-1)
                if buckets is None:
                    buckets = paddle.argmax(rotated_vectors_factor, axis=-1)
                else:
                    buckets = buckets + (cur_product * paddle.argmax(
                        rotated_vectors_factor, axis=-1))

                cur_product = cur_product * bucket_factor

        if attention_mask is not None and (
                attention_mask.sum() < batch_size * attention_mask.shape[-1]):
            # add an extra bucket for padding tokens only
            num_buckets = num_buckets + 1
            # assign padding tokens extra bucket
            buckets_mask = attention_mask.unsqueeze(
                axis=[1, 2]).expand_as(buckets)
            buckets = paddle.where(
                buckets_mask.astype(paddle.bool),
                buckets,
                paddle.to_tensor(
                    num_buckets - 1, dtype=buckets.dtype), )
        elif increase_num_buckets:
            num_buckets = num_buckets + 1

        # buckets is now (Batch_size x Num_Attn_Heads x Num_Hashes x Seq_Len).
        # Next we add offsets so that bucket numbers from different hashing rounds don't overlap.
        offsets = paddle.arange(num_hashes)
        offsets = (offsets * num_buckets).reshape(shape=[1, 1, -1, 1])

        # expand to batch size and num attention heads
        offsets = offsets.expand(
            shape=[batch_size, self.num_attention_heads] + offsets.shape[-2:])
        offset_buckets = (buckets + offsets).flatten(start_axis=2, stop_axis=3)

        return offset_buckets

    def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(self, buckets):
        # no gradients are needed
        # buckets shape [batch_size, self.num_attention_heads, num_hashes * sequence_length]
        with paddle.no_grad():
            original_shape = buckets.shape
            new_buckets = buckets.flatten(0, 1)
            offsets = (paddle.arange(new_buckets.shape[0]) *
                       new_buckets.shape[1]).unsqueeze(-1)
            sorted_bucket_idx = _stable_argsort(new_buckets, axis=-1)
            new_sorted_bucket_idx = (sorted_bucket_idx + offsets).flatten()
            updates = paddle.tile(
                paddle.arange(new_buckets.shape[1]),
                repeat_times=[new_buckets.shape[0]])

            undo_sorted_bucket_idx = paddle.scatter(
                paddle.zeros_like(new_sorted_bucket_idx),
                new_sorted_bucket_idx,
                updates,
                overwrite=True, )

        return sorted_bucket_idx.reshape(
            shape=original_shape), undo_sorted_bucket_idx.reshape(
                shape=original_shape)

    def _set_num_buckets(self, sequence_length):
        # `num_buckets` should be set to 2 * sequence_length // chunk_length as recommended in paper
        num_buckets_pow_2 = (2 * (sequence_length // self.chunk_length)
                             ).bit_length() - 1
        # make sure buckets are power of 2
        num_buckets = 2**num_buckets_pow_2

        # factorize `num_buckets` if `num_buckets` becomes too large
        num_buckets_limit = 2 * max(
            int((self.max_position_embeddings // self.chunk_length)**(0.5)),
            self.chunk_length, )
        if num_buckets > num_buckets_limit:
            num_buckets = [
                2**(num_buckets_pow_2 // 2),
                2**(num_buckets_pow_2 - num_buckets_pow_2 // 2),
            ]

        logger.warning(
            f"num_buckets is not set. Setting num_buckets to {num_buckets}...")

        # set num buckets in config to be properly saved
        self.num_buckets = num_buckets

    def _attend(
            self,
            query_vectors,
            key_vectors,
            value_vectors,
            sorted_bucket_idx_per_hash,
            attention_mask,
            do_standard_self_attention,
            do_cached_attention, ):
        # look at previous and following chunks if chunked attention
        if not do_standard_self_attention:
            key_vectors = self._look_adjacent(
                key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors = self._look_adjacent(
                value_vectors, self.num_chunks_before, self.num_chunks_after)

        # get logits and dots
        # (BS, NumAttn, NumHash x NumChunk, Chunk_L x Hidden),(BS, NumAttn, NumHash x NumChunk, Chunk_L * (Num_bef + Num_aft + 1) x Hidden) -> (BS, NumAttn, NumHash x NumChunk, Chunk_L, Chunk_L * (1 + Num_bef + Num_aft))
        query_key_dots = paddle.matmul(
            query_vectors, key_vectors, transpose_y=True)

        # free memory
        del query_vectors, key_vectors

        # if chunked attention split bucket idxs to query and key
        if not do_standard_self_attention:
            query_bucket_idx = self._split_seq_length_dim_to(
                sorted_bucket_idx_per_hash,
                -1,
                self.chunk_length,
                self.num_attention_heads, )
            key_value_bucket_idx = self._look_adjacent(
                query_bucket_idx, self.num_chunks_before, self.num_chunks_after)
        elif do_cached_attention and query_key_dots.ndim > 4:
            key_value_bucket_idx = sorted_bucket_idx_per_hash
            query_bucket_idx = (paddle.ones(
                shape=key_value_bucket_idx.shape[:-1] + [1],
                dtype=key_value_bucket_idx.dtype,
            ) * key_value_bucket_idx.max())
        elif do_cached_attention and query_key_dots.ndim <= 4:
            query_bucket_idx = (
                query_key_dots.shape[-1] - 1
            ) * paddle.ones_like(query_key_dots)[:, :, :, -1]
            key_value_bucket_idx = (paddle.arange(query_key_dots.shape[-1])
                                    .unsqueeze(axis=[0, 1])
                                    .expand(shape=query_bucket_idx.shape[:2] +
                                            [query_key_dots.shape[-1], ]))
        else:
            query_bucket_idx = key_value_bucket_idx = sorted_bucket_idx_per_hash

        # get correct mask values depending on precision
        if query_key_dots.dtype == paddle.float16:
            self_mask_value = self.self_mask_value_float16.astype(
                paddle.float16)
            mask_value = self.mask_value_float16.astype(paddle.float16)
        else:
            self_mask_value = self.self_mask_value_float32
            mask_value = self.mask_value_float32

        if not do_cached_attention:
            mask = self._compute_attn_mask(
                query_bucket_idx,
                key_value_bucket_idx,
                attention_mask,
                query_key_dots.shape,
                do_standard_self_attention, )

            if mask is not None:
                query_key_dots = paddle.where(
                    mask.astype(paddle.bool), query_key_dots, mask_value)

            # free memory
            del mask

        # Self mask is ALWAYS applied.
        # From the reformer paper (https://arxiv.org/pdf/2001.04451.pdf):
        # " While attention to the future is not allowed, typical implementations of the
        # Transformer do allow a position to attend to itself.
        # Such behavior is undesirable in a shared-QK formulation because the dot-product
        # of a query vector with itself will almost always be greater than the dot product of a
        # query vector with a vector at another position. We therefore modify the masking
        # to forbid a token from attending to itself, except in situations
        # where a token has no other valid attention targets (e.g. the first token in a sequence) "
        self_mask = paddle.not_equal(
            query_bucket_idx.unsqueeze(-1).astype("int64"),
            key_value_bucket_idx.unsqueeze(-2).astype("int64"))

        # apply self_mask
        query_key_dots = paddle.where(self_mask, query_key_dots,
                                      self_mask_value)

        # free memory
        del self_mask

        logits = _logsumexp(query_key_dots, axis=-1, keepdim=True)
        # dots shape is `[batch_size, num_attn_heads, num_hashes * seq_len // chunk_length, chunk_length, chunk_length * (1 + num_chunks_before + num_chunks_after)]`
        attention_probs = paddle.exp(query_key_dots - logits)

        # free memory
        del query_key_dots

        # dropout
        attention_probs = F.dropout(
            attention_probs, p=self.dropout, training=self.training)

        # attend values
        out_vectors = paddle.matmul(attention_probs, value_vectors)

        # free memory
        del value_vectors

        # merge chunk length
        if out_vectors.ndim > 4:

            logits = logits.flatten(start_axis=2, stop_axis=3).squeeze(-1)
            out_vectors = out_vectors.flatten(start_axis=2, stop_axis=3)

        return out_vectors, logits, attention_probs

    def _compute_attn_mask(
            self,
            query_indices,
            key_indices,
            attention_mask,
            query_key_dot_shape,
            do_standard_self_attention, ):

        # attention mask for LSH
        if attention_mask is not None:
            # if chunked attention, the attention mask has to correspond to LSH order
            attention_mask = attention_mask.astype(paddle.int64).unsqueeze(1)
            if not do_standard_self_attention:
                # expand attn_mask to fit with key_value_bucket_idx shape
                attention_mask = attention_mask.unsqueeze(1)
                attention_mask = attention_mask.expand(
                    shape=query_indices.shape[:-1] +
                    [attention_mask.shape[-1]])

                attention_mask = attention_mask.reshape(
                    [-1, attention_mask.shape[-1]])
                new_key_indices = key_indices.reshape(
                    [-1, key_indices.shape[-1]])
                attention_mask = paddle.index_sample(
                    attention_mask, new_key_indices).reshape(key_indices.shape)

            attention_mask = attention_mask.unsqueeze(-2).expand(
                shape=query_key_dot_shape)

        # Causal mask
        if self.is_decoder is True:

            causal_mask = paddle.greater_equal(
                query_indices.unsqueeze(-1),
                key_indices.unsqueeze(-2)).astype(paddle.int64)

            # add attention mask if not None
            if attention_mask is not None:
                attention_mask = causal_mask * attention_mask
            else:
                attention_mask = causal_mask

        return attention_mask

    def _get_relevant_hid_states_and_buckets(
            self,
            query_vectors,
            attention_mask,
            num_hashes,
            hidden_states,
            past_states,
            past_buckets, ):
        # concat hidden states
        hidden_states = paddle.concat([past_states, hidden_states], axis=1)

        # batch_size hidden
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]

        # check if cached buckets include pad bucket
        max_bucket = (self.num_buckets if isinstance(self.num_buckets, int) else
                      reduce(mul, self.num_buckets))

        # if pad bucket was cached => need to increase num buckets for caching
        increase_num_buckets = past_buckets.max() > num_hashes * max_bucket - 1

        # retrieve query buckets
        query_buckets = self._hash_vectors(
            query_vectors,
            num_hashes,
            attention_mask,
            increase_num_buckets=increase_num_buckets, )

        # concat buckets
        concat_buckets = paddle.concat(
            [past_buckets, query_buckets.unsqueeze(-1)], axis=-1)

        # hash-based sort
        bucket_idx = paddle.argsort(concat_buckets, axis=-1)

        # bucket_idx has shape: BatchSize x NumAttnHeads x NumHashes x SequenceLength
        assert bucket_idx.shape == [
            batch_size,
            self.num_attention_heads,
            num_hashes,
            sequence_length,
        ], f"bucket_idx should have shape {(batch_size, self.num_attention_heads, num_hashes, sequence_length)}, but has shape {bucket_idx.shape}."

        # find indices of new bucket indices
        relevant_bucket_idx = (bucket_idx ==
                               (bucket_idx.shape[-1] - 1)).nonzero()

        # expand relevant bucket indices to its chunks
        relevant_bucket_idx_chunk = self._expand_to_indices_in_relevant_chunk(
            relevant_bucket_idx, sequence_length)

        relevant_bucket_idx_chunk = bucket_idx.gather_nd(
            relevant_bucket_idx_chunk)

        # adapt bucket_idx for batch and hidden states for index select
        bucket_idx_batch_offset = sequence_length * (
            batch_size * paddle.arange(relevant_bucket_idx_chunk.shape[-1]) //
            relevant_bucket_idx_chunk.shape[-1])

        # add batch offset
        relevant_bucket_idx_chunk_all_batch = (
            relevant_bucket_idx_chunk + bucket_idx_batch_offset)
        hidden_states = hidden_states.reshape(shape=[-1, self.hidden_size])

        # select all relevant hidden states
        relevant_hidden_states = hidden_states.index_select(
            relevant_bucket_idx_chunk_all_batch, axis=0)

        # reshape hidden states and bucket_idx to correct output
        relevant_hidden_states = relevant_hidden_states.reshape(
            shape=[batch_size, self.num_attention_heads, -1, self.hidden_size])
        relevant_bucket_idx_chunk = relevant_bucket_idx_chunk.reshape(
            shape=[batch_size, self.num_attention_heads, num_hashes, -1])

        assert (
            relevant_hidden_states.shape[2] ==
            (self.num_chunks_before + self.num_chunks_after + 1
             ) * self.chunk_length * num_hashes
        ), f"There should be {(self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length * num_hashes} `hidden_states`, there are {relevant_hidden_states.shape[2]} `hidden_states`."

        assert (
            relevant_bucket_idx_chunk.shape[-1] ==
            (self.num_chunks_before + self.num_chunks_after + 1
             ) * self.chunk_length
        ), f"There should be {(self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length} `hidden_states`, there are {relevant_bucket_idx_chunk.shape[-1]} `bucket_idx`."

        return relevant_hidden_states, relevant_bucket_idx_chunk, query_buckets

    def _expand_to_indices_in_relevant_chunk(self, indices, sequence_length):
        # get relevant indices of where chunk starts and its size
        start_indices_chunk = (
            (indices[:, -1] // self.chunk_length) - self.num_chunks_before
        ) * self.chunk_length
        total_chunk_size = self.chunk_length * (
            1 + self.num_chunks_before + self.num_chunks_after)

        # expand start indices and add correct chunk offset via arange
        expanded_start_indices = start_indices_chunk.unsqueeze(-1).expand(
            shape=[indices.shape[0], total_chunk_size])
        chunk_sequence_indices = expanded_start_indices + paddle.arange(
            total_chunk_size).unsqueeze(0).expand(
                shape=[indices.shape[0], total_chunk_size])

        # make sure that circular logic holds via % seq len
        chunk_sequence_indices = chunk_sequence_indices.flatten(
        ) % sequence_length

        # expand indices and set indices correctly
        indices = (indices.unsqueeze(1).expand(shape=(
            indices.shape[0], total_chunk_size, indices.shape[-1]))
                   .flatten(0, 1).clone())
        indices[:, -1] = chunk_sequence_indices

        return indices

    def _len_and_dim_norm(self, vectors):
        """
        length and attention head size dim normalization
        """
        vectors = self._len_norm(vectors)
        vectors = vectors * paddle.rsqrt(
            paddle.to_tensor(
                self.attention_head_size, dtype=vectors.dtype))
        return vectors

    def _len_norm(self, x, epsilon=1e-6):
        """
        length normalization
        """
        variance = paddle.mean(x**2, axis=-1, keepdim=True)
        norm_x = x * paddle.rsqrt(variance + epsilon)
        return norm_x

    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        """
        expand dims of idxs and vectors for all hashes and gather
        """
        expanded_idxs = paddle.tile(
            idxs.unsqueeze(-2),
            repeat_times=[1, 1, self.attention_head_size, 1]).reshape(
                shape=[-1, idxs.shape[2]])
        vectors = (paddle.tile(
            vectors,
            repeat_times=[1, 1, num_hashes, 1]).transpose(perm=[0, 1, 3, 2])
                   .reshape(shape=[-1, idxs.shape[2]]))

        return (paddle.index_sample(vectors, expanded_idxs).reshape(
            shape=[idxs.shape[0], idxs.shape[1], self.attention_head_size, -1])
                .transpose(perm=[0, 1, 3, 2]))


class LocalSelfAttention(nn.Layer, EfficientAttentionMixin):
    def __init__(
            self,
            num_attention_heads=8,
            local_attn_chunk_length=128,
            local_num_chunks_before=1,
            local_num_chunks_after=0,
            is_decoder=True,
            pad_token_id=0,
            attention_head_size=128,
            hidden_size=1024,
            local_attention_probs_dropout_prob=0.2, ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.chunk_length = local_attn_chunk_length
        self.num_chunks_before = local_num_chunks_before
        self.num_chunks_after = local_num_chunks_after
        self.is_decoder = is_decoder
        self.pad_token_id = pad_token_id

        self.attention_head_size = attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = hidden_size

        # projection matrices
        self.query = nn.Linear(
            self.hidden_size, self.all_head_size, bias_attr=False)
        self.key = nn.Linear(
            self.hidden_size, self.all_head_size, bias_attr=False)
        self.value = nn.Linear(
            self.hidden_size, self.all_head_size, bias_attr=False)

        self.dropout = local_attention_probs_dropout_prob

        # save mask value here
        self.register_buffer("mask_value_float16", paddle.to_tensor(-1e4))
        self.register_buffer("mask_value_float32", paddle.to_tensor(-1e9))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            cache=None,
            use_cache=False,
            output_attentions=False,
            **kwargs, ):
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        # check if cache shall be used and that hidden states are already cached
        if use_cache and cache[1] is not None:
            assert (
                cache[0] is None
            ), "LocalSelfAttention should not make use of `buckets`. There seems to be an error when caching hidden_states_and_buckets."
            key_value_hidden_states = self._retrieve_relevant_hidden_states(
                cache[1], self.chunk_length, self.num_chunks_before)
            key_value_hidden_states = paddle.concat(
                [key_value_hidden_states, hidden_states], axis=1)

            # only query vector for last token
            query_vectors = self.query(hidden_states)
            # compute key and value for relevant chunk
            key_vectors = self.key(key_value_hidden_states)
            value_vectors = self.value(key_value_hidden_states)

            # free memory
            del key_value_hidden_states
        else:
            # project hidden_states to query, key and value
            query_vectors = self.query(hidden_states)
            key_vectors = self.key(hidden_states)
            value_vectors = self.value(hidden_states)

        # split last dim into `num_attention_heads` and `attention_head_size`
        query_vectors = self._split_hidden_size_dim(
            query_vectors, self.num_attention_heads, self.attention_head_size)
        key_vectors = self._split_hidden_size_dim(
            key_vectors, self.num_attention_heads, self.attention_head_size)
        value_vectors = self._split_hidden_size_dim(
            value_vectors, self.num_attention_heads, self.attention_head_size)

        assert (
            query_vectors.shape[-1] == self.attention_head_size
        ), f"last dim of query_key_vectors is {query_vectors.shape[-1]} but should be {self.attention_head_size}."
        assert (
            key_vectors.shape[-1] == self.attention_head_size
        ), f"last dim of query_key_vectors is {key_vectors.shape[-1]} but should be {self.attention_head_size}."
        assert (
            value_vectors.shape[-1] == self.attention_head_size
        ), f"last dim of query_key_vectors is {value_vectors.shape[-1]} but should be {self.attention_head_size}."

        if self.chunk_length is None:
            assert (
                self.num_chunks_before == 0 and self.num_chunks_after == 0
            ), "If `chunk_length` is `None`, make sure `num_chunks_after` and `num_chunks_before` are set to 0."

        # normalize key vectors
        key_vectors = key_vectors / paddle.sqrt(
            paddle.to_tensor(
                self.attention_head_size, dtype=key_vectors.dtype))

        # get sequence length indices
        indices = paddle.tile(
            paddle.arange(sequence_length),
            repeat_times=[batch_size, self.num_attention_heads, 1], )

        # if one should do normal n^2 self-attention
        do_standard_self_attention = sequence_length <= self.chunk_length

        # if input should be chunked
        if not do_standard_self_attention:
            # chunk vectors

            # B x Num_Attn_Head x Seq_Len // chunk_len x chunk_len  x  attn_head_size

            query_vectors = self._split_seq_length_dim_to(
                query_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size, )

            key_vectors = self._split_seq_length_dim_to(
                key_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size, )
            value_vectors = self._split_seq_length_dim_to(
                value_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size, )

            query_indices = self._split_seq_length_dim_to(
                indices, -1, self.chunk_length, self.num_attention_heads)
            key_indices = self._split_seq_length_dim_to(
                indices, -1, self.chunk_length, self.num_attention_heads)

            # append chunks before and after
            key_vectors = self._look_adjacent(
                key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors = self._look_adjacent(
                value_vectors, self.num_chunks_before, self.num_chunks_after)
            key_indices = self._look_adjacent(
                key_indices, self.num_chunks_before, self.num_chunks_after)
        else:
            query_indices = key_indices = indices

        # query-key matmul: QK^T
        query_key_dots = paddle.matmul(
            query_vectors, key_vectors, transpose_y=True)

        # free memory
        del query_vectors, key_vectors

        mask = self._compute_attn_mask(
            query_indices,
            key_indices,
            attention_mask,
            query_key_dots.shape,
            do_standard_self_attention, )

        if mask is not None:
            # get mask tensor depending on half precision or not
            if query_key_dots.dtype == paddle.float16:
                mask_value = self.mask_value_float16.astype(paddle.float16)
            else:
                mask_value = self.mask_value_float32

            query_key_dots = paddle.where(
                mask.astype(paddle.bool), query_key_dots, mask_value)

        # free memory
        del mask

        # softmax
        logits = _logsumexp(query_key_dots, axis=-1, keepdim=True)
        attention_probs = paddle.exp(query_key_dots - logits)

        # free memory
        del logits

        # dropout
        attention_probs = F.dropout(
            attention_probs, p=self.dropout, training=self.training)

        # attend values
        out_vectors = paddle.matmul(attention_probs, value_vectors)

        # free memory
        del value_vectors

        # merge chunk length
        if not do_standard_self_attention:
            out_vectors = out_vectors.flatten(start_axis=2, stop_axis=3)

        assert out_vectors.shape == [
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        ]
        out_vectors = self._merge_hidden_size_dims(
            out_vectors, self.num_attention_heads, self.attention_head_size)

        if output_attentions is False:
            attention_probs = ()

        return LocalSelfAttentionOutput(
            hidden_states=out_vectors, attention_probs=attention_probs)

    def _compute_attn_mask(
            self,
            query_indices,
            key_indices,
            attention_mask,
            query_key_dots_shape,
            do_standard_self_attention, ):

        # chunk attention mask and look before and after

        if attention_mask is not None:

            attention_mask = attention_mask.astype(paddle.int64).unsqueeze(1)

            if not do_standard_self_attention:
                attention_mask = self._split_seq_length_dim_to(
                    attention_mask, -1, self.chunk_length, 1)
                attention_mask = self._look_adjacent(attention_mask,
                                                     self.num_chunks_before,
                                                     self.num_chunks_after)
            # create attn_mask

            attention_mask = attention_mask.unsqueeze(-2).expand(
                shape=query_key_dots_shape)

        # Causal mask
        if self.is_decoder is True:
            causal_mask = paddle.greater_equal(
                query_indices.unsqueeze(-1),
                key_indices.unsqueeze(-2)).astype(paddle.int64)

            # add attention mask if not None
            if attention_mask is not None:
                attention_mask = causal_mask * attention_mask
            else:
                attention_mask = causal_mask

        return attention_mask

    @staticmethod
    def _retrieve_relevant_hidden_states(previous_hidden_states, chunk_length,
                                         num_chunks_before):
        start_position = ((previous_hidden_states.shape[1] // chunk_length) -
                          num_chunks_before) * chunk_length
        return previous_hidden_states[:, start_position:]


class ReformerSelfOutput(nn.Layer):
    def __init__(
            self,
            num_attention_heads=8,
            attention_head_size=128,
            hidden_dropout_prob=0.2,
            hidden_size=1024, ):
        super().__init__()
        all_head_size = num_attention_heads * attention_head_size
        self.dropout = hidden_dropout_prob

        self.dense = nn.Linear(all_head_size, hidden_size, bias_attr=False)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.dropout(
            hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class ReformerAttention(nn.Layer):
    def __init__(
            self,
            attn_layers=[
                "local",
                "local",
                "lsh",
                "local",
                "local",
                "local",
                "lsh",
                "local",
                "local",
                "local",
                "lsh",
                "local",
            ],
            hidden_size=1024,
            layer_norm_eps=1e-12,
            lsh_attn_chunk_length=256,
            num_hashes=4,
            num_buckets=512,
            lsh_num_chunks_before=1,
            lsh_num_chunks_after=0,
            hash_seed=None,
            is_decoder=True,
            max_position_embeddings=65536,
            lsh_attention_probs_dropout_prob=0.1,
            num_attention_heads=8,
            attention_head_size=128,
            local_attn_chunk_length=128,
            local_num_chunks_before=1,
            local_num_chunks_after=0,
            pad_token_id=0,
            local_attention_probs_dropout_prob=0.2,
            hidden_dropout_prob=0.2,
            layer_id=0, ):
        super().__init__()
        self.layer_id = layer_id
        self.attn_layers = attn_layers

        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)

        if len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "lsh":
            self.self_attention = LSHSelfAttention(
                lsh_attn_chunk_length=lsh_attn_chunk_length,
                num_hashes=num_hashes,
                num_buckets=num_buckets,
                lsh_num_chunks_before=lsh_num_chunks_before,
                lsh_num_chunks_after=lsh_num_chunks_after,
                hash_seed=hash_seed,
                is_decoder=is_decoder,
                max_position_embeddings=max_position_embeddings,
                lsh_attention_probs_dropout_prob=lsh_attention_probs_dropout_prob,
                num_attention_heads=num_attention_heads,
                attention_head_size=attention_head_size,
                hidden_size=hidden_size, )
        elif len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "local":
            self.self_attention = LocalSelfAttention(
                num_attention_heads=num_attention_heads,
                local_attn_chunk_length=local_attn_chunk_length,
                local_num_chunks_before=local_num_chunks_before,
                local_num_chunks_after=local_num_chunks_after,
                is_decoder=is_decoder,
                pad_token_id=pad_token_id,
                attention_head_size=attention_head_size,
                hidden_size=hidden_size,
                local_attention_probs_dropout_prob=local_attention_probs_dropout_prob,
            )
        elif len(set(self.attn_layers)) == 2 and set(self.attn_layers) == set(
            ["lsh", "local"]):
            # get correct attn layers
            if self.attn_layers[self.layer_id] == "lsh":
                self.self_attention = LSHSelfAttention(
                    lsh_attn_chunk_length=lsh_attn_chunk_length,
                    num_hashes=num_hashes,
                    num_buckets=num_buckets,
                    lsh_num_chunks_before=lsh_num_chunks_before,
                    lsh_num_chunks_after=lsh_num_chunks_after,
                    hash_seed=hash_seed,
                    is_decoder=is_decoder,
                    max_position_embeddings=max_position_embeddings,
                    lsh_attention_probs_dropout_prob=lsh_attention_probs_dropout_prob,
                    num_attention_heads=num_attention_heads,
                    attention_head_size=attention_head_size,
                    hidden_size=hidden_size, )
            else:
                self.self_attention = LocalSelfAttention(
                    num_attention_heads=num_attention_heads,
                    local_attn_chunk_length=local_attn_chunk_length,
                    local_num_chunks_before=local_num_chunks_before,
                    local_num_chunks_after=local_num_chunks_after,
                    is_decoder=is_decoder,
                    pad_token_id=pad_token_id,
                    attention_head_size=attention_head_size,
                    hidden_size=hidden_size,
                    local_attention_probs_dropout_prob=local_attention_probs_dropout_prob,
                )
        else:
            raise NotImplementedError(
                f"Only attn layer types 'lsh' and 'local' exist, but got `attn_layers`: {self.attn_layers}. "
                "Select attn layer types from ['lsh', 'local'] only.")
        self.output = ReformerSelfOutput(
            num_attention_heads=num_attention_heads,
            attention_head_size=attention_head_size,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_size=hidden_size, )

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            num_hashes=None,
            cache=None,
            use_cache=False,
            orig_sequence_length=None,
            output_attentions=False,
            buckets=None, ):
        hidden_states = self.layer_norm(hidden_states)

        # make sure cached hidden states is set to None for backward pass
        if cache is not None:
            cache_layer = cache[self.layer_id]
        else:
            cache_layer = None

        # use cached buckets for backprob if buckets not None for LSHSelfAttention
        self_attention_outputs = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            cache=cache_layer,
            use_cache=use_cache,
            output_attentions=output_attentions,
            buckets=buckets, )

        # add buckets if necessary
        if hasattr(self_attention_outputs, "buckets"):
            buckets = self_attention_outputs.buckets
        else:
            buckets = None

        # cache hidden states for future use
        if use_cache:
            if cache[self.layer_id][0] is None:
                # padded input should not be cached
                past_buckets = (
                    buckets[:, :, :, :orig_sequence_length]
                    if (buckets is not None and orig_sequence_length > 1) else
                    buckets)
            else:
                past_buckets = paddle.concat(
                    [cache[self.layer_id][0], buckets], axis=-1)

            if cache[self.layer_id][1] is None:
                # padded input should not be cached
                past_states = hidden_states[:, :orig_sequence_length]
            else:
                past_states = paddle.concat(
                    [cache[self.layer_id][1], hidden_states], axis=1)

            cache[self.layer_id] = (past_buckets, past_states)
        # compute attention feed forward output
        attention_output = self.output(self_attention_outputs.hidden_states)

        return AttentionOutput(
            hidden_states=attention_output,
            attention_probs=self_attention_outputs.attention_probs,
            buckets=buckets, )


class ReformerFeedForwardDense(nn.Layer):
    def __init__(
            self,
            hidden_dropout_prob=0.2,
            hidden_act="relu",
            hidden_size=1024,
            feed_forward_size=4096, ):
        super().__init__()
        self.dropout = hidden_dropout_prob

        if isinstance(hidden_act, str):
            self.act_fn = ACT2FN[hidden_act]
        else:
            self.act_fn = hidden_act

        self.dense = nn.Linear(hidden_size, feed_forward_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.dropout(
            hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class ReformerFeedForwardOutput(nn.Layer):
    def __init__(self,
                 hidden_dropout_prob=0.2,
                 feed_forward_size=4096,
                 hidden_size=1024):
        super().__init__()
        self.dropout = hidden_dropout_prob

        self.dense = nn.Linear(feed_forward_size, hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.dropout(
            hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class ChunkReformerFeedForward(nn.Layer):
    def __init__(
            self,
            chunk_size_feed_forward=0,
            hidden_size=1024,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            hidden_act="relu",
            feed_forward_size=4096, ):
        super().__init__()
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.seq_len_dim = 1

        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dense = ReformerFeedForwardDense(
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            hidden_size=hidden_size,
            feed_forward_size=feed_forward_size, )
        self.output = ReformerFeedForwardOutput(
            hidden_dropout_prob=hidden_dropout_prob,
            feed_forward_size=feed_forward_size,
            hidden_size=hidden_size, )

    def forward(self, attention_output):
        return _apply_chunking_to_forward(
            self.forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output, )

    def forward_chunk(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        return self.output(hidden_states)


class ReformerLayer(nn.Layer):
    def __init__(
            self,
            attn_layers=[
                "local",
                "local",
                "lsh",
                "local",
                "local",
                "local",
                "lsh",
                "local",
                "local",
                "local",
                "lsh",
                "local",
            ],
            hidden_size=1024,
            layer_norm_eps=1e-12,
            lsh_attn_chunk_length=256,
            num_hashes=4,
            num_buckets=512,
            lsh_num_chunks_before=1,
            lsh_num_chunks_after=0,
            hash_seed=None,
            is_decoder=True,
            max_position_embeddings=65536,
            lsh_attention_probs_dropout_prob=0.1,
            num_attention_heads=8,
            attention_head_size=128,
            local_attn_chunk_length=128,
            local_num_chunks_before=1,
            local_num_chunks_after=0,
            pad_token_id=0,
            local_attention_probs_dropout_prob=0.2,
            hidden_dropout_prob=0.2,
            chunk_size_feed_forward=0,
            hidden_act="relu",
            feed_forward_size=4096,
            layer_id=0, ):
        super().__init__()
        self.attention = ReformerAttention(
            attn_layers=attn_layers,
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            lsh_attn_chunk_length=lsh_attn_chunk_length,
            num_hashes=num_hashes,
            num_buckets=num_buckets,
            lsh_num_chunks_before=lsh_num_chunks_before,
            lsh_num_chunks_after=lsh_num_chunks_after,
            hash_seed=hash_seed,
            is_decoder=is_decoder,
            max_position_embeddings=max_position_embeddings,
            lsh_attention_probs_dropout_prob=lsh_attention_probs_dropout_prob,
            num_attention_heads=num_attention_heads,
            attention_head_size=attention_head_size,
            local_attn_chunk_length=local_attn_chunk_length,
            local_num_chunks_before=local_num_chunks_before,
            local_num_chunks_after=local_num_chunks_after,
            pad_token_id=pad_token_id,
            local_attention_probs_dropout_prob=local_attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_id=layer_id, )
        # dropout requires to have the same
        # seed for forward and backward pass
        self.attention_seed = None
        self.feed_forward_seed = None

        self.feed_forward = ChunkReformerFeedForward(
            chunk_size_feed_forward=chunk_size_feed_forward,
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            feed_forward_size=feed_forward_size, )

    def _init_attention_seed(self):
        """
        This function sets a new seed for the attention layer to make dropout 
        deterministic for both forward calls: 1 normal forward call and 1 forward 
        call in backward to recalculate activations.
        """

        # randomize seeds
        # use cuda generator if available
        if paddle.get_device() != "cpu":
            # GPU
            device_idx = int(paddle.get_device().split(":")[1])
            sts = paddle.get_cuda_rng_state()
            self.attention_seed = sts[device_idx].current_seed()
        else:
            # CPU
            self.attention_seed = np.random.randint(
                0, sys.maxsize, size=(1, ), dtype="int64").item()

        paddle.seed(self.attention_seed)

    def _init_feed_forward_seed(self):
        """
        This function sets a new seed for the feed forward layer to make dropout deterministic for both forward calls:
        1 normal forward call and 1 forward call in backward to recalculate activations.
        """
        # randomize seeds
        # use cuda generator if available
        if paddle.get_device() != "cpu":
            # GPU
            device_idx = int(paddle.get_device().split(":")[1])
            sts = paddle.get_cuda_rng_state()
            self.feed_forward_seed = sts[device_idx].current_seed()
        else:
            # CPU
            self.feed_forward_seed = np.random.randint(
                0, sys.maxsize, size=(1, ), dtype="int64").item()

        paddle.seed(self.feed_forward_seed)

    def forward(
            self,
            prev_attn_output,
            hidden_states,
            attention_mask=None,
            num_hashes=None,
            cache=None,
            use_cache=False,
            orig_sequence_length=None,
            output_attentions=False, ):
        with paddle.no_grad():
            # every forward pass we sample a different seed
            # for dropout and save for forward fn in backward pass
            # to have correct dropout
            if self.training:
                self._init_attention_seed()

            attn_outputs = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                num_hashes=num_hashes,
                cache=cache,
                use_cache=use_cache,
                orig_sequence_length=orig_sequence_length,
                output_attentions=output_attentions, )
            attn_output = attn_outputs.hidden_states

            # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
            # Y_1 = X_1 + f(X_2)
            attn_output = prev_attn_output + attn_output

            # free memory
            del prev_attn_output

            # every forward pass we sample a different seed
            # for dropout and save seed for forward fn in backward
            # to have correct dropout
            if self.training:
                self._init_feed_forward_seed()
            # Y_2 = X_2 + g(Y_1)
            hidden_states = hidden_states + self.feed_forward(attn_output)

        return ReformerOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            attention_probs=attn_outputs.attention_probs,
            buckets=attn_outputs.buckets, )

    def backward_pass(
            self,
            next_attn_output,
            hidden_states,
            grad_attn_output,
            grad_hidden_states,
            attention_mask=None,
            buckets=None, ):
        # Implements the backward pass for reversible ResNets.
        # A good blog post on how this works can be found here:
        # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
        # This code is heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py

        assert (
            self.training
        ), "If you want to train `ReformerModel` and its variations, make sure to use `model.train()` to put the model into training mode."

        with paddle.set_grad_enabled(True):
            next_attn_output.stop_gradient = False
            # set seed to have correct dropout
            paddle.seed(self.feed_forward_seed)
            # g(Y_1)
            res_hidden_states = self.feed_forward(next_attn_output)
            res_hidden_states.backward(grad_hidden_states, retain_graph=True)

        with paddle.no_grad():
            # X_2 = Y_2 - g(Y_1)
            hidden_states = hidden_states - res_hidden_states
            del res_hidden_states
            grad_attn_output = grad_attn_output + next_attn_output.grad

            next_attn_output.stop_gradient = True

        with paddle.set_grad_enabled(True):
            hidden_states.stop_gradient = False

            # set seed to have correct dropout
            paddle.seed(self.attention_seed)
            # f(X_2)
            # use cached buckets for backprob if buckets not None for LSHSelfAttention
            output = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                buckets=buckets, ).hidden_states
            output.backward(grad_attn_output, retain_graph=True)

        with paddle.no_grad():
            # X_1 = Y_1 - f(X_2)
            attn_output = next_attn_output - output
            del output, next_attn_output

            grad_hidden_states = grad_hidden_states + hidden_states.grad
            hidden_states.stop_gradient = True
            hidden_states = hidden_states.detach()

        return ReformerBackwardOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            grad_attn_output=grad_attn_output,
            grad_hidden_states=grad_hidden_states, )


class ReformerEncoder(nn.Layer):
    def __init__(
            self,
            num_hidden_layers=12,
            attn_layers=[
                "local",
                "local",
                "lsh",
                "local",
                "local",
                "local",
                "lsh",
                "local",
                "local",
                "local",
                "lsh",
                "local",
            ],
            hidden_size=1024,
            layer_norm_eps=1e-12,
            lsh_attn_chunk_length=256,
            num_hashes=4,
            num_buckets=512,
            lsh_num_chunks_before=1,
            lsh_num_chunks_after=0,
            hash_seed=None,
            is_decoder=True,
            max_position_embeddings=65536,
            lsh_attention_probs_dropout_prob=0.1,
            num_attention_heads=8,
            attention_head_size=128,
            local_attn_chunk_length=128,
            local_num_chunks_before=1,
            local_num_chunks_after=0,
            pad_token_id=0,
            local_attention_probs_dropout_prob=0.2,
            hidden_dropout_prob=0.2,
            chunk_size_feed_forward=0,
            hidden_act="relu",
            feed_forward_size=4096, ):
        super().__init__()
        self.dropout = hidden_dropout_prob

        self.layers = nn.LayerList([
            ReformerLayer(
                attn_layers=attn_layers,
                hidden_size=hidden_size,
                layer_norm_eps=layer_norm_eps,
                lsh_attn_chunk_length=lsh_attn_chunk_length,
                num_hashes=num_hashes,
                num_buckets=num_buckets,
                lsh_num_chunks_before=lsh_num_chunks_before,
                lsh_num_chunks_after=lsh_num_chunks_after,
                hash_seed=hash_seed,
                is_decoder=is_decoder,
                max_position_embeddings=max_position_embeddings,
                lsh_attention_probs_dropout_prob=lsh_attention_probs_dropout_prob,
                num_attention_heads=num_attention_heads,
                attention_head_size=attention_head_size,
                local_attn_chunk_length=local_attn_chunk_length,
                local_num_chunks_before=local_num_chunks_before,
                local_num_chunks_after=local_num_chunks_after,
                pad_token_id=pad_token_id,
                local_attention_probs_dropout_prob=local_attention_probs_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                chunk_size_feed_forward=chunk_size_feed_forward,
                hidden_act=hidden_act,
                feed_forward_size=feed_forward_size,
                layer_id=i, ) for i in range(num_hidden_layers)
        ])
        # Reformer is using Rev Nets, thus last layer outputs are concatenated and
        # Layer Norm is done over 2 * hidden_size
        self.layer_norm = nn.LayerNorm(2 * hidden_size, epsilon=layer_norm_eps)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            num_hashes=None,
            cache=None,
            use_cache=False,
            orig_sequence_length=None,
            output_hidden_states=False,
            output_attentions=False, ):
        # hidden_states and attention lists to be filled if wished
        all_hidden_states = []
        all_attentions = []

        # init cached hidden states if necessary
        if cache is None:
            cache = [((None), (None)) for i in range(len(self.layers))]

        # concat same tensor for reversible ResNet
        hidden_states = paddle.concat([hidden_states, hidden_states], axis=-1)
        hidden_states = _ReversibleFunction.apply(
            hidden_states,
            self.layers,
            attention_mask,
            num_hashes,
            all_hidden_states,
            all_attentions,
            cache,
            use_cache,
            orig_sequence_length,
            output_hidden_states,
            output_attentions, )

        # Apply layer norm to concatenated hidden states
        hidden_states = self.layer_norm(hidden_states)

        # Apply dropout
        hidden_states = F.dropout(
            hidden_states, p=self.dropout, training=self.training)

        return ReformerEncoderOutput(
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_attentions=all_attentions,
            cache=cache, )


class ReformerOnlyLMHead(nn.Layer):
    def __init__(self, chunk_size_lm_head=0, hidden_size=1024, vocab_size=258):
        super().__init__()
        # Reformer is using Rev Nets, thus last layer outputs are concatenated and
        # Layer Norm is done over 2 * hidden_size
        self.seq_len_dim = 1
        self.chunk_size_lm_head = chunk_size_lm_head
        self.decoder = nn.Linear(2 * hidden_size, vocab_size, bias_attr=False)
        self.bias = self.create_parameter(shape=(vocab_size, ), is_bias=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        return _apply_chunking_to_forward(self.forward_chunk,
                                          self.chunk_size_lm_head,
                                          self.seq_len_dim, hidden_states)

    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ReformerClassificationHead(nn.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, classifier_dropout, num_classes):
        super().__init__()
        self.dense = nn.Linear(2 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        hidden_states = hidden_states[:, 0]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = F.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class ReformerPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained Reformer models. It provides Reformer related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    base_model_prefix = "reformer"
    model_config_file = "model_config.json"

    pretrained_init_configuration = {
        "reformer-enwik8": {
            "tie_word_embeddings": False,
            "is_decoder": True,
            "chunk_size_feed_forward": 0,
            "pad_token_id": 0,
            "hash_seed": None,
            "vocab_size": 258,
            "attention_head_size": 128,
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_hashes": 4,
            "num_hidden_layers": 12,
            "num_buckets": 512,
            "lsh_attn_chunk_length": 256,
            "local_attn_chunk_length": 128,
            "lsh_num_chunks_after": 0,
            "lsh_num_chunks_before": 1,
            "local_num_chunks_after": 0,
            "local_num_chunks_before": 1,
            "hidden_act": "relu",
            "feed_forward_size": 4096,
            "hidden_dropout_prob": 0.2,
            "lsh_attention_probs_dropout_prob": 0.1,
            "local_attention_probs_dropout_prob": 0.2,
            "max_position_embeddings": 65536,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "axial_pos_embds": True,
            "axial_pos_shape": [128, 512],
            "axial_pos_embds_dim": [256, 768],
            "axial_norm_std": 1.0,
            "chunk_size_lm_head": 0,
            "attn_layers": [
                "local",
                "local",
                "lsh",
                "local",
                "local",
                "local",
                "lsh",
                "local",
                "local",
                "local",
                "lsh",
                "local",
            ],
        },
        "reformer-crime-and-punishment": {
            "tie_word_embeddings": False,
            "is_decoder": True,
            "chunk_size_feed_forward": 0,
            "pad_token_id": 0,
            "num_hidden_layers": 6,
            "hash_seed": None,
            "vocab_size": 320,
            "attention_head_size": 64,
            "hidden_size": 256,
            "num_attention_heads": 2,
            "num_hashes": 1,
            "num_buckets": [64, 128],
            "lsh_attn_chunk_length": 64,
            "local_attn_chunk_length": 64,
            "lsh_num_chunks_after": 0,
            "lsh_num_chunks_before": 1,
            "local_num_chunks_after": 0,
            "local_num_chunks_before": 1,
            "hidden_act": "relu",
            "feed_forward_size": 512,
            "hidden_dropout_prob": 0.05,
            "lsh_attention_probs_dropout_prob": 0.0,
            "local_attention_probs_dropout_prob": 0.05,
            "max_position_embeddings": 524288,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "axial_pos_embds": True,
            "axial_pos_shape": [512, 1024],
            "axial_pos_embds_dim": [64, 192],
            "axial_norm_std": 1.0,
            "chunk_size_lm_head": 0,
            "attn_layers": ["local", "lsh", "local", "lsh", "local", "lsh"],
        },
    }

    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "reformer-enwik8":
            "http://paddlenlp.bj.bcebos.com/models/transformers/reformer/reformer-enwik8/model_state.pdparams",
            "reformer-crime-and-punishment":
            "http://paddlenlp.bj.bcebos.com/models/transformers/reformer/reformer-crime-and-punishment/model_state.pdparams",
        }
    }

    def init_weights(self):
        """
        Initializes and tie weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)
        # Tie weights if needed
        self.tie_weights()

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        tie_word_embeddings = (
            self.tie_word_embeddings if hasattr(self, "tie_word_embeddings")
            else self.reformer.config.get("tie_word_embeddings", False))
        if (hasattr(self, "get_output_embeddings") and
                hasattr(self, "get_input_embeddings") and tie_word_embeddings):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings,
                                           self.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone layer weights"""
        if output_embeddings.weight.shape == input_embeddings.weight.shape:
            output_embeddings.weight = input_embeddings.weight
        elif output_embeddings.weight.shape == input_embeddings.weight.t(
        ).shape:
            output_embeddings.weight.set_value(input_embeddings.weight.t())
        else:
            raise ValueError(
                "when tie input/output embeddings, the shape of output embeddings: {}"
                "should be equal to shape of input embeddings: {}"
                "or should be equal to the shape of transpose input embeddings: {}".
                format(
                    output_embeddings.weight.shape,
                    input_embeddings.weight.shape,
                    input_embeddings.weight.t().shape, ))
        if getattr(output_embeddings, "bias", None) is not None:
            if output_embeddings.weight.shape[
                    -1] != output_embeddings.bias.shape[0]:
                raise ValueError(
                    "the weight lase shape: {} of output_embeddings is not equal to the bias shape: {}"
                    "please check output_embeddings configuration".format(
                        output_embeddings.weight.shape[-1],
                        output_embeddings.bias.shape[0], ))

    def _init_weights(self, layer):
        """Initialize the weights"""
        if isinstance(layer, AxialPositionEmbeddings):
            for weight in layer.weights:
                weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.axial_norm_std
                        if hasattr(self, "axial_norm_std") else
                        self.reformer.config["axial_norm_std"],
                        shape=weight.shape, ))

        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.reformer.config["initializer_range"],
                    shape=layer.weight.shape, ))

        elif isinstance(layer, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.axial_norm_std if hasattr(self, "axial_norm_std")
                    else self.reformer.config["axial_norm_std"],
                    shape=layer.weight.shape, ))

            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))

        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.full_like(layer.weight, 1.0))


@register_base_model
class ReformerModel(ReformerPretrainedModel):
    """
    The bare Reformer Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        tie_word_embeddings (bool, optional):
            Whether to tie input and output embeddings. Defaults to `False`.
        is_decoder (bool, optional):
            Whether or not to use a causal mask in addition to the `attention_mask` passed to `ReformerModel`. When using the Reformer for causal language modeling, this argument should be set to `True`. Defaults to `True`.
        chunk_size_feed_forward (int, optional):
            The chunk size of all feed forward layers in the residual attention blocks. A chunk size of `0` means
            that the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes
            `n` < sequence_length embeddings at a time. Defaults to `0`.
        pad_token_id (int, optional):
            The id of the `padding` token. Defaults to `0`.
        hash_seed (int, optional):
            Seed that can be used to make local sensitive hashing in `LSHSelfAttention` deterministic. This should
            only be set for testing purposed. For evaluation and training purposes `hash_seed` should be left as
            `None` to ensure fully random rotations in local sensitive hashing scheme. Defaults to `None`.
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in `ReformerModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `ReformerModel`. Defaults to `258`.
        attention_head_size (int, optional):
            Dimensionality of the projected key, query and value vectors. Defaults to `128`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layer.Defaults to `1024`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `8`.
        num_hashes (int, optional):
            Number of hashing rounds (e.g., number of random rotations) in Local Sensitive Hashing scheme. The higher `num_hashes`, the more accurate the `LSHSelfAttention` becomes, but also the more memory and time intensive the hashing becomes. Defaults to `4`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        num_buckets (int or List[int], optional):
            Number of buckets, the key query vectors can be "hashed into" using the locality sensitive hashing scheme.
            Each query key vector is hashed into a hash in `1, ..., num_buckets`. The number of buckets can also be factorized into a list for improved memory complexity. In this case, each query key vector is hashed into a hash in `1-1, 1-2, ..., num_buckets[0]-1, ..., num_buckets[0]-num_buckets[1]` if `num_buckets` is factorized into two factors. The number of buckets (or the product the factors) should approximately equal sequence length / lsh_chunk_length. If `num_buckets` not set, a good value is calculated on the fly. Defaults to `512`.
        lsh_attn_chunk_length (int, optional):
            Length of chunk which attends to itself in `LSHSelfAttention`. Chunking reduces memory complexity from sequence length x sequence length (self attention) to chunk length x chunk length x sequence length / chunk length (chunked self attention).Defaults to `256`.
        local_attn_chunk_length (int, optional):
            Length of chunk which attends to itself in `LocalSelfAttention`. Chunking reduces memory complexity from sequence length x sequence length (self attention) to chunk length x chunk length x sequence length / chunk length (chunked self attention).Defaults to `128`.
        lsh_num_chunks_after (int, optional):
            Number of following neighbouring chunks to attend to in `LSHSelfAttention` layer to itself. Defaults to `0`.
        lsh_num_chunks_before (int, optional):
            Number of previous neighbouring chunks to attend to in `LSHSelfAttention` layer to itself. Defaults to `1`.
        local_num_chunks_after (int, optional):
            Number of following neighbouring chunks to attend to in `LocalSelfAttention` layer to itself. Defaults to `0`.
        local_num_chunks_before (int, optional):
            Number of previous neighbouring chunks to attend to in `LocalSelfAttention` layer to itself. Defaults to `1`.
        hidden_act (str, optional):
            The non-linear activation function (function or string) in the feed forward layer in the residual attention block. If string, `"gelu"`, `"relu"`, `"tanh"`, `"mish"` and `"gelu_new"` are supported. Defaults to `"relu"`.
        feed_forward_size (int, optional):
            Dimensionality of the feed_forward layer in the residual attention block. Defaults to `4096`.
        hidden_dropout_prob (float, optional):
            The dropout ratio for all fully connected layers in the embeddings and encoder. Defaults to `0.2`.
        lsh_attention_probs_dropout_prob (float, optional):
            The dropout ratio for the attention probabilities in `LSHSelfAttention`. Defaults to `0.1`.
        local_attention_probs_dropout_prob (float, optional):
            The dropout ratio for the attention probabilities in `LocalSelfAttention`. Defaults to `0.2`.
        max_position_embeddings (int, optional):
            The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048). Defaults to `65536`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to `0.02`.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`ReformerPretrainedModel._init_weights()` for how weights are initialized in `ReformerModel`.

        layer_norm_eps (float, optional):
            The epsilon used by the layer normalization layers. Defaults to `1e-12`.

        axial_pos_embds (bool, optional):
            Whether or not to use axial position embeddings. Defaults to `True`.
        axial_pos_shape (List[int], optional):
            The position dims of the axial position encodings. During training, the product of the position dims has to be equal to the sequence length. Defaults to `[128, 512]`.
        axial_pos_embds_dim (List[int], optional):
            The embedding dims of the axial position encodings. The sum of the embedding dims has to be equal to the
            hidden size. Defaults to `[256, 768]`.
        axial_norm_std (float, optional):
            The standard deviation of the normal_initializer for initializing the weight matrices of the axial
            positional encodings. Defaults to `1.0`.
        chunk_size_lm_head (int, optional):
            The chunk size of the final language model feed forward head layer. A chunk size of 0 means that the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes n <
            sequence_length embeddings at a time. Defaults to `0`.
        attn_layers (List[str], optional):
            List of attention layer types in ascending order. It can be chosen between a LSHSelfAttention layer
            (`"lsh"`) and a LocalSelfAttention layer (`"local"`). Defaults to `["local", "local", "lsh", "local", "local", "local", "lsh", "local", "local", "local", "lsh", "local"]`.

    """

    def __init__(
            self,
            tie_word_embeddings=False,
            is_decoder=True,
            chunk_size_feed_forward=0,
            pad_token_id=0,
            hash_seed=None,
            vocab_size=258,
            attention_head_size=128,
            hidden_size=1024,
            num_attention_heads=8,
            num_hashes=4,
            num_hidden_layers=12,
            num_buckets=512,
            lsh_attn_chunk_length=256,
            local_attn_chunk_length=128,
            lsh_num_chunks_after=0,
            lsh_num_chunks_before=1,
            local_num_chunks_after=0,
            local_num_chunks_before=1,
            hidden_act="relu",
            feed_forward_size=4096,
            hidden_dropout_prob=0.2,
            lsh_attention_probs_dropout_prob=0.1,
            local_attention_probs_dropout_prob=0.2,
            max_position_embeddings=65536,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            axial_pos_embds=True,
            axial_pos_shape=[128, 512],
            axial_pos_embds_dim=[256, 768],
            axial_norm_std=1.0,
            chunk_size_lm_head=0,
            attn_layers=[
                "local",
                "local",
                "lsh",
                "local",
                "local",
                "local",
                "lsh",
                "local",
                "local",
                "local",
                "lsh",
                "local",
            ], ):
        super().__init__()
        assert (
            num_hidden_layers > 0
        ), "`attn_layers` is empty. Select at least one attn layer form ['lsh', 'local']"
        self.initializer_range = initializer_range
        self.axial_norm_std = axial_norm_std
        self.tie_word_embeddings = tie_word_embeddings
        self.chunk_size_lm_head = chunk_size_lm_head
        self.pad_token_id = pad_token_id
        self.attn_layers = attn_layers
        self.lsh_attn_chunk_length = lsh_attn_chunk_length
        self.local_attn_chunk_length = local_attn_chunk_length

        self.embeddings = ReformerEmbeddings(
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            axial_pos_embds=axial_pos_embds,
            axial_pos_shape=axial_pos_shape,
            axial_pos_embds_dim=axial_pos_embds_dim,
            attn_layers=attn_layers,
            lsh_attn_chunk_length=lsh_attn_chunk_length,
            local_attn_chunk_length=local_attn_chunk_length, )
        self.encoder = ReformerEncoder(
            hidden_dropout_prob=hidden_dropout_prob,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            attn_layers=attn_layers,
            lsh_attn_chunk_length=lsh_attn_chunk_length,
            num_hashes=num_hashes,
            num_buckets=num_buckets,
            lsh_num_chunks_before=lsh_num_chunks_before,
            lsh_num_chunks_after=lsh_num_chunks_after,
            hash_seed=hash_seed,
            is_decoder=is_decoder,
            max_position_embeddings=max_position_embeddings,
            lsh_attention_probs_dropout_prob=lsh_attention_probs_dropout_prob,
            num_attention_heads=num_attention_heads,
            attention_head_size=attention_head_size,
            local_attn_chunk_length=local_attn_chunk_length,
            local_num_chunks_before=local_num_chunks_before,
            local_num_chunks_after=local_num_chunks_after,
            pad_token_id=pad_token_id,
            local_attention_probs_dropout_prob=local_attention_probs_dropout_prob,
            hidden_act=hidden_act,
            feed_forward_size=feed_forward_size,
            chunk_size_feed_forward=chunk_size_feed_forward, )

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            num_hashes=None,
            cache=None,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False, ):
        r"""
        The ReformerModel forward method, overrides the `__call__()` special method.

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
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            position_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. 
                Selected in the range `[0, max_position_embeddings - 1]`.
                Shape as [batch_size, num_tokens] and dtype as int64. Defaults to `None`.
            num_hashes (int, optional):
                The number of hashing rounds that should be performed during bucketing. Setting 
                this argument overwrites the default defined in `config["num_hashes"]`. 
                Defaults to `None`.
            cache (List[tuple(Tensor, Tensor)], optional):
                List of `tuple(Tensor, Tensor)` of length `config["num_hidden_layers"]`, with 
                the first element being the previous `buckets` of shape `[batch_size, num_heads, num_hashes, sequence_length]` and the second being the previous `hidden_states` of shape 
                `[batch_size, sequence_length, hidden_size]`.
                Contains precomputed hidden-states and buckets (only relevant for LSH Self-Attention). Can 
                be used to speed up sequential decoding. 
                Defaults to `None`.
            use_cache (bool, optional):
                Whether or not to use cache. If set to `True`, `cache` states are returned 
                and can be used to speed up decoding. 
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether or not to return the attentions tensors of all attention layers.
                Defaults to `False`.
            output_hidden_states (bool, optional):
                Whether or not to return the output of all hidden layers.
                Defaults to `False`.

        Returns:
            tuple: Returns tuple (`last_hidden_state`, `cache`, `hidden_states`, `attentions`)

            With the fields:

            - `last_hidden_state` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and 
                its shape is [batch_size, sequence_length, hidden_size].

            - `cache` (List[tuple(Tensor, Tensor)], optional):
                returned when `use_cache=True` is passed.
                List of `tuple(Tensor, Tensor)` of length `config["num_hidden_layers"]`, 
                with the first element being the previous `buckets` of shape 
                `[batch_size, num_heads, num_hashes, sequence_length]` and the second 
                being the previous `hidden_states` of shape `[batch_size, sequence_length, hidden_size]`.

            - `hidden_states` (tuple(Tensor), optional):
                returned when `output_hidden_states=True` is passed.
                tuple of `Tensor` (one for the output of the embeddings + one for the 
                output of each layer). Each Tensor has a data type of float32 
                and its shape is [batch_size, sequence_length, hidden_size].
                
            - `attentions` (tuple(Tensor), optional):
                returned when `output_attentions=True` is passed.
                tuple of `Tensor` (one for each layer) of shape. Each Tensor has a data 
                type of float32 and its shape is [batch_size, num_heads, sequence_length, sequence_length].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ReformerModel, ReformerTokenizer

                tokenizer = ReformerTokenizer.from_pretrained('reformer-crime-and-punishment')
                model = ReformerModel.from_pretrained('reformer-crime-and-punishment')
                model.eval()

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                outputs = model(**inputs)
                last_hidden_state = outputs[0]

        """

        input_shape = input_ids.shape

        assert (
            len(input_shape) == 2
        ), f"`input_ids` have be of shape `[batch_size, sequence_length]`, but got shape: {input_shape}"

        if cache is not None:
            assert (
                not self.training
            ), "`cache` can only be used for inference, not for training`."

        # original sequence length for padding
        orig_sequence_length = input_shape[-1]

        # if needs padding
        least_common_mult_chunk_length = _get_least_common_mult_chunk_len(
            self.attn_layers, self.lsh_attn_chunk_length,
            self.local_attn_chunk_length)
        min_chunk_length = _get_min_chunk_len(self.attn_layers,
                                              self.lsh_attn_chunk_length,
                                              self.local_attn_chunk_length)

        must_pad_to_match_chunk_length = (
            input_shape[-1] % least_common_mult_chunk_length != 0 and
            input_shape[-1] > min_chunk_length and cache is None)

        if must_pad_to_match_chunk_length:
            padding_length = (least_common_mult_chunk_length - input_shape[-1] %
                              least_common_mult_chunk_length)

            if self.training is True:
                raise ValueError(
                    f"If training, sequence length {input_shape[-1]} has to be a multiple of least common multiple "
                    f"chunk_length {least_common_mult_chunk_length}. Please consider padding the input to a length "
                    f"of {input_shape[-1] + padding_length}.")

            # pad input
            (
                input_ids,
                attention_mask,
                position_ids,
                input_shape, ) = self._pad_to_mult_of_chunk_length(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    input_shape=input_shape,
                    padding_length=padding_length,
                    padded_seq_length=least_common_mult_chunk_length, )

        # start index for position encoding depends on incremental decoding
        if cache is not None:
            start_idx_pos_encodings = cache[0][1].shape[1]
        else:
            start_idx_pos_encodings = 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            start_idx_pos_encodings=start_idx_pos_encodings, )

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            cache=cache,
            use_cache=use_cache,
            orig_sequence_length=orig_sequence_length,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions, )
        sequence_output = encoder_outputs.hidden_states

        # if padding was applied
        if must_pad_to_match_chunk_length:
            sequence_output = sequence_output[:, :orig_sequence_length]

        cache = encoder_outputs.cache if use_cache else None
        hidden_states = (encoder_outputs.all_hidden_states
                         if output_hidden_states else None)
        attentions = encoder_outputs.all_attentions if output_attentions else None

        return tuple(v
                     for v in [
                         sequence_output,
                         cache,
                         hidden_states,
                         attentions,
                     ] if v is not None)

    def _pad_to_mult_of_chunk_length(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            input_shape=None,
            padding_length=None,
            padded_seq_length=None, ):
        logger.info(
            f"Input ids are automatically padded from {input_shape[-1]} to {input_shape[-1] + padding_length} to be a "
            f"multiple of `chunk_length`: {padded_seq_length}")

        padded_input_ids = paddle.full(
            (input_shape[0], padding_length),
            self.pad_token_id,
            dtype=paddle.int64, )

        # Extend `attention_mask`
        if attention_mask is not None:
            pad_attention_mask = paddle.zeros(
                shape=[input_shape[0], padding_length],
                dtype=attention_mask.dtype)

            attention_mask = paddle.concat(
                [attention_mask, pad_attention_mask], axis=-1)
        else:
            attention_mask = paddle.concat(
                [
                    paddle.ones(
                        input_shape, dtype=paddle.int64),
                    paddle.zeros(
                        shape=(input_shape[0], padding_length),
                        dtype=paddle.int64),
                ],
                axis=-1, )

        input_ids = paddle.concat([input_ids, padded_input_ids], axis=-1)
        input_shape = input_ids.shape

        # Pad position ids if given
        if position_ids is not None:
            padded_position_ids = paddle.arange(input_shape[-1],
                                                padded_seq_length)
            padded_position_ids = position_ids.unsqueeze(0).expand(
                shape=[input_shape[0], padding_length])
            position_ids = paddle.concat(
                [position_ids, padded_position_ids], axis=-1)

        return input_ids, attention_mask, position_ids, input_shape


class ReformerModelWithLMHead(ReformerPretrainedModel):
    """
    The Reformer Model transformer with a language modeling head on top.

    Args:
        reformer (:class:`ReformerModel`):
            An instance of :class:`ReformerModel`.

    """

    def __init__(self, reformer):
        super().__init__()
        self.reformer = reformer
        local_num_chunks_after = self.reformer.config["local_num_chunks_after"]
        lsh_num_chunks_after = self.reformer.config["lsh_num_chunks_after"]
        assert self.reformer.config[
            "is_decoder"], "If you want to use `ReformerModelWithLMHead` make sure that `is_decoder=True`."
        assert (
            "local" not in self.reformer.config["attn_layers"] or
            local_num_chunks_after == 0
        ), f"If causal mask is enabled, make sure that `local_num_chunks_after` is set to 0 and not {local_num_chunks_after}."
        assert (
            "lsh" not in self.reformer.config["attn_layers"] or
            lsh_num_chunks_after == 0
        ), f"If causal mask is enabled, make sure that `lsh_num_chunks_after` is set to 1 and not {lsh_num_chunks_after}."

        self.lm_head = ReformerOnlyLMHead(
            chunk_size_lm_head=self.reformer.config["chunk_size_lm_head"],
            hidden_size=self.reformer.config["hidden_size"],
            vocab_size=self.reformer.config["vocab_size"], )

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            num_hashes=None,
            cache=None,
            use_cache=False,
            labels=None,
            output_hidden_states=False,
            output_attentions=False, ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`ReformerModel`.
            position_ids (Tensor, optional):
                See :class:`ReformerModel`.
            attention_mask (Tensor, optional):
                See :class:`ReformerModel`.
            num_hashes (int, optional):
                See :class:`ReformerModel`.
            cache (List[tuple(Tensor, Tensor)], optional):
                See :class:`ReformerModel`.
            use_cache (bool, optional):
                See :class:`ReformerModel`.
            labels (Tensor, optional):
                Labels for language modeling. Note that the labels **are shifted** 
                inside the model, i.e. you can set `labels = input_ids` Indices are 
                selected in `[-100, 0, ..., vocab_size]` All labels set to `-100` are 
                ignored (masked), the loss is only computed for labels in `[0, ..., vocab_size]`.
                Shape is [batch_size, sequence_length] and dtype is int64.
            output_attentions (bool, optional):
                See :class:`ReformerModel`.
            output_hidden_states (bool, optional):
                See :class:`ReformerModel`.

        Returns:
            tuple: Returns tuple `(loss, logits, cache, hidden_states, attentions)`.

            With the fields:

            - `loss` (Tensor):
                returned when `labels` is provided.
                Language modeling loss (for next-token prediction).
                It's data type should be float32 and its shape is [1,].

            - `logits` (Tensor):
                Prediction scores of the language modeling head 
                (scores for each vocabulary token before SoftMax).
                It's data type should be float32 and its shape is 
                [batch_size, sequence_length, vocab_size].

            - `cache` (List[tuple(Tensor, Tensor)]):
                See :class:`ReformerModel`.

            - `hidden_states` (tuple(Tensor)):
                See :class:`ReformerModel`.

            - `attentions` (tuple(Tensor)):
                See :class:`ReformerModel`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ReformerModelWithLMHead, ReformerTokenizer

                tokenizer = ReformerTokenizer.from_pretrained('reformer-crime-and-punishment')
                model = ReformerModelWithLMHead.from_pretrained('reformer-crime-and-punishment')
                model.eval()

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs, labels=inputs["input_ids"])

                loss = output[0]
                logits = output[1]

        """

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            cache=cache,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions, )

        sequence_output = reformer_outputs[0]
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(
                    shape=[-1, self.reformer.config["vocab_size"]]),
                shift_labels.flatten(), )

        output = (logits, ) + reformer_outputs[1:]
        return ((loss, ) + output) if loss is not None else output

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      cache=None,
                                      use_cache=None,
                                      num_hashes=None,
                                      **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        if cache is not None:
            input_ids = input_ids[:, -1:]

        inputs_dict = {
            "input_ids": input_ids,
            "cache": cache,
            "use_cache": use_cache,
            "num_hashes": num_hashes,
        }

        return inputs_dict


class ReformerForMaskedLM(ReformerPretrainedModel):
    """
    The Reformer Model transformer with a masked language modeling head on top.

    Args:
        reformer (:class:`ReformerModel`):
            An instance of :class:`ReformerModel`.

    """

    def __init__(self, reformer):
        super().__init__()
        self.reformer = reformer
        assert not self.reformer.config[
            "is_decoder"], "If you want to use `ReformerForMaskedLM` make sure `is_decoder=False` for bi-directional self-attention."
        self.lm_head = ReformerOnlyLMHead(
            chunk_size_lm_head=self.reformer.config["chunk_size_lm_head"],
            hidden_size=self.reformer.config["hidden_size"],
            vocab_size=self.reformer.config["vocab_size"], )

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            num_hashes=None,
            labels=None,
            output_hidden_states=False,
            output_attentions=False, ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`ReformerModel`.
            position_ids (Tensor, optional):
                See :class:`ReformerModel`.
            attention_mask (Tensor, optional):
                See :class:`ReformerModel`.
            num_hashes (int, optional):
                See :class:`ReformerModel`.
            labels (Tensor, optional):
                Labels for computing the masked language modeling loss. 
                Indices should be in ``[-100, 0, ..., vocab_size]`` 
                (see ``input_ids`` docstring) Tokens with indices set 
                to ``-100`` are ignored(masked), the loss is only computed 
                for the tokens with labels in ``[0, ..., vocab_size]``.
                Shape is [batch_size, sequence_length] and dtype is int64.
            output_attentions (bool, optional):
                See :class:`ReformerModel`.
            output_hidden_states (bool, optional):
                See :class:`ReformerModel`.

        Returns:
            tuple: Returns tuple `(loss, logits, hidden_states, attentions)`.

            With the fields:

            - `loss` (Tensor):
                returned when `labels` is provided.
                Masked Language modeling loss.
                It's data type should be float32 and its shape is [1,].

            - `logits` (Tensor):
                Prediction scores of the masked language modeling head 
                (scores for each vocabulary token before SoftMax).
                It's data type should be float32 and its shape is 
                [batch_size, sequence_length, vocab_size].

            - `hidden_states` (tuple(Tensor)):
                See :class:`ReformerModel`.

            - `attentions` (tuple(Tensor)):
                See :class:`ReformerModel`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ReformerForMaskedLM, ReformerTokenizer

                tokenizer = ReformerTokenizer.from_pretrained('reformer-crime-and-punishment')
                model = ReformerForMaskedLM.from_pretrained('reformer-crime-and-punishment', is_decoder=False)
                model.eval()

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs, labels=inputs["input_ids"])

                loss = output[0]
                logits = output[1]

        """

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            use_cache=False,  # no causal mask
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions, )

        sequence_output = reformer_outputs[0]
        logits = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                logits.reshape(shape=[-1, self.reformer.config["vocab_size"]]),
                labels.flatten(), )

        output = (logits, ) + reformer_outputs[1:]
        return ((masked_lm_loss, ) + output
                ) if masked_lm_loss is not None else output


class ReformerForSequenceClassification(ReformerPretrainedModel):
    """
    The Reformer Model transformer with a sequence classification head on top (linear layer).

    Args:
        reformer (:class:`ReformerModel`):
            An instance of :class:`ReformerModel`.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of Reformer.
            If None, use the same value as `hidden_dropout_prob` of `ReformerModel`
            instance `reformer`. Defaults to None.

    """

    def __init__(self, reformer, num_classes=2, dropout=None):
        super().__init__()
        self.reformer = reformer
        self.num_classes = num_classes
        classifier_dropout = (dropout if dropout is not None else
                              self.reformer.config["hidden_dropout_prob"])
        self.classifier = ReformerClassificationHead(
            hidden_size=self.reformer.config["hidden_size"],
            classifier_dropout=classifier_dropout,
            num_classes=num_classes, )
        if self.reformer.config["is_decoder"] is True:
            logger.warning(
                "You might want to disable causal masking for sequence classification"
            )

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            num_hashes=None,
            labels=None,
            output_hidden_states=False,
            output_attentions=False, ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`ReformerModel`.
            position_ids (Tensor, optional):
                See :class:`ReformerModel`.
            attention_mask (Tensor, optional):
                See :class:`ReformerModel`.
            num_hashes (int, optional):
                See :class:`ReformerModel`.
            labels (Tensor, optional):
                Labels for computing the sequence classification/regression loss. Indices 
                should be in `[0, ...,num_classes - 1]`. If `num_classes == 1` a regression 
                loss is computed (Mean-Square loss), If `num_classes > 1` a classification 
                loss is computed (Cross-Entropy).
                Shape is [batch_size,] and dtype is int64.
            output_attentions (bool, optional):
                See :class:`ReformerModel`.
            output_hidden_states (bool, optional):
                See :class:`ReformerModel`.

        Returns:
            tuple: Returns tuple `(loss, logits, hidden_states, attentions)`.

            With the fields:

            - `loss` (Tensor):
                returned when `labels` is provided.
                Classification (or regression if num_classes==1) loss.
                It's data type should be float32 and its shape is [1,].

            - `logits` (Tensor):
                Classification (or regression if num_classes==1) scores (before SoftMax).
                It's data type should be float32 and its shape is [batch_size, num_classes].

            - `hidden_states` (tuple(Tensor)):
                See :class:`ReformerModel`.

            - `attentions` (tuple(Tensor)):
                See :class:`ReformerModel`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ReformerForSequenceClassification, ReformerTokenizer

                tokenizer = ReformerTokenizer.from_pretrained('reformer-crime-and-punishment')
                model = ReformerForSequenceClassification.from_pretrained('reformer-crime-and-punishment', is_decoder=False)
                model.eval()

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs, labels=paddle.to_tensor([0]))

                loss = output[0]
                logits = output[1]

        """

        outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions, )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_classes == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.flatten(),
                                labels.astype(logits.dtype).flatten())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.reshape([-1, self.num_classes]), labels.flatten())

        output = (logits, ) + outputs[1:]
        return ((loss, ) + output) if loss is not None else output


class ReformerForQuestionAnswering(ReformerPretrainedModel):
    """
    Reformer Model with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and
    `span end logits`).

    Args:
        reformer (:class:`ReformerModel`):
            An instance of ReformerModel.
        dropout (float, optional):
            The dropout probability for output of Reformer.
            If None, use the same value as `hidden_dropout_prob` of `ReformerModel` instance `reformer`. Defaults to `None`.

    """

    def __init__(self, reformer, dropout=None):
        super().__init__()
        self.reformer = reformer
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.reformer.config["hidden_dropout_prob"])
        # 2 * hidden_size because we use reversible residual layers
        self.qa_outputs = nn.Linear(2 * self.reformer.config["hidden_size"], 2)
        if self.reformer.config["is_decoder"] is True:
            logger.warning(
                "You might want to disable causal masking for question answering task."
            )
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            num_hashes=None,
            start_positions=None,
            end_positions=None,
            output_hidden_states=False,
            output_attentions=False, ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`ReformerModel`.
            position_ids (Tensor, optional):
                See :class:`ReformerModel`.
            attention_mask (Tensor, optional):
                See :class:`ReformerModel`.
            num_hashes (int, optional):
                See :class:`ReformerModel`.
            start_positions (Tensor, optional):
                Labels for position (index) of the start of the labelled 
                span for computing the token classification loss.
                Positions are clamped to the length of the sequence 
                (`sequence_length`). Position outside of the sequence 
                are not taken into account for computing the loss.
                Shape is [batch_size,] and dtype is int64.
            end_positions (Tensor, optional):
                Labels for position (index) of the end of the labelled 
                span for computing the token classification loss.
                Positions are clamped to the length of the sequence 
                (`sequence_length`). Position outside of the sequence 
                are not taken into account for computing the loss.
                Shape is [batch_size,] and dtype is int64.
            output_attentions (bool, optional):
                See :class:`ReformerModel`.
            output_hidden_states (bool, optional):
                See :class:`ReformerModel`.

        Returns:
            tuple: Returns tuple `(loss, logits, hidden_states, attentions)`.

            With the fields:

            - `loss` (Tensor):
                returned when `labels` is provided.
                Classification (or regression if num_classes==1) loss.
                It's data type should be float32 and its shape is [1,].

            - `start_logits` (Tensor):
                A tensor of the input token classification logits, indicates 
                the start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `end_logits` (Tensor):
                A tensor of the input token classification logits, indicates 
                the end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `hidden_states` (tuple(Tensor)):
                See :class:`ReformerModel`.

            - `attentions` (tuple(Tensor)):
                See :class:`ReformerModel`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ReformerForQuestionAnswering, ReformerTokenizer

                tokenizer = ReformerTokenizer.from_pretrained('reformer-crime-and-punishment')
                model = ReformerForQuestionAnswering.from_pretrained('reformer-crime-and-punishment', is_decoder=False)
                model.eval()

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)

                start_logits = outputs[0]
                end_logits = outputs[1]

        """

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            use_cache=False,  # no causal mask
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions, )

        sequence_output = reformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(2, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) + reformer_outputs[1:]
        return ((total_loss, ) + output) if total_loss is not None else output
