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

import math
import re
import numpy as np
import inspect
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import CrossEntropyLoss
from paddle import Tensor
from collections import OrderedDict
from typing import Optional, Tuple, Union, List, Dict, Any
from dataclasses import dataclass, fields, is_dataclass
from ppocr.modeling.backbones.rec_donut_swin import DonutSwinModelOutput
from ppocr.modeling.heads.rec_unimernet_head import (
    MBartForCausalLM,
    MBartDecoder,
    MBartConfig,
    ModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    zeros_,
    ones_,
    kaiming_normal_,
    trunc_normal_,
    xavier_uniform_,
    CausalLMOutputWithCrossAttentions,
    LogitsProcessorList,
    ForcedEOSTokenLogitsProcessor,
    UniMERNetHead,
)


@dataclass
class AttentionMaskConverter:
    """
    A class to convert attention masks based on specific configurations.

    This class is designed to handle the conversion of attention masks with options for causal masking
    and sliding window attention, which are commonly used in transformer models.

    Attributes:
        is_causal (bool): Flag indicating whether the attention mask should enforce causal masking,
                          which ensures each position can only attend to previous positions.
        sliding_window (int, optional): Size of the sliding window for local attention. If set,
                                        attention is restricted to a local window of this size.

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
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        if is_export:
            mask = paddle.full(
                (tgt_len, tgt_len), paddle.finfo(dtype).min, dtype="float64"
            )
            mask_cond = paddle.arange(mask.shape[-1])
            mask.masked_fill_(
                mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1]), 0
            )
        else:
            mask = paddle.full((tgt_len, tgt_len), paddle.finfo(dtype).min)
            mask_cond = paddle.arange(mask.shape[-1])
            mask.masked_fill_(
                mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1]), 0
            )
            mask = mask.cast(dtype)

        if past_key_values_length > 0:
            mask = paddle.concat(
                [paddle.zeros(tgt_len, past_key_values_length, dtype=dtype), mask],
                axis=-1,
            )

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1

            context_mask = paddle.tril(
                paddle.ones_like(mask, dtype=paddle.bool), diagonal=diagonal
            )
            mask.masked_fill_(context_mask, paddle.finfo(dtype).min)

        return mask[None, None, :, :].expand(
            [bsz, 1, tgt_len, tgt_len + past_key_values_length]
        )

    @staticmethod
    def _make_causal_mask_parallel(
        input_ids_shape,
        dtype,
        past_key_values_length=0,
        sliding_window=None,
        parallel_step=1,
        is_export=False,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = paddle.full((tgt_len, tgt_len), paddle.finfo(dtype).min)
        mask_cond = paddle.arange(mask.shape[-1])
        mask_cond_parallel = paddle.arange(mask.shape[-1])

        mask_parallel = paddle.arange(0, tgt_len, step=parallel_step).reshape([1, -1])
        mask_parallel = paddle.repeat_interleave(mask_parallel, parallel_step, 1)[
            :, :tgt_len
        ]
        mask.masked_fill_(
            mask_cond < (mask_parallel + parallel_step).reshape([mask.shape[-1], 1]), 0
        )
        mask = mask.cast(dtype)

        if past_key_values_length > 0:
            mask = paddle.concat(
                [paddle.zeros([tgt_len, past_key_values_length], dtype=dtype), mask],
                axis=-1,
            )

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1

            context_mask = paddle.tril(
                paddle.ones_like(mask, dtype=paddle.bool), diagonal=diagonal
            )
            mask.masked_fill_(context_mask, paddle.finfo(dtype).min)

        return mask[None, None, :, :].expand(
            [bsz, 1, tgt_len, tgt_len + past_key_values_length]
        )

    def to_4d(
        self,
        attention_mask_2d,
        query_length,
        dtype,
        key_value_length,
        use_parallel=False,
        parallel_step=3,
        is_export=False,
    ):
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        input_shape = (attention_mask_2d.shape[0], query_length)

        causal_4d_mask = None
        if use_parallel:
            step = parallel_step
        else:
            step = 1
        if (
            input_shape[-1] > step or self.sliding_window is not None
        ) and self.is_causal:

            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length

            if use_parallel:
                causal_4d_mask = self._make_causal_mask_parallel(
                    input_shape,
                    dtype,
                    past_key_values_length=past_key_values_length,
                    sliding_window=self.sliding_window,
                    parallel_step=parallel_step,
                    is_export=is_export,
                )
            else:
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
            expanded_attn_mask = causal_4d_mask.masked_fill_(
                expanded_attn_mask.cast(paddle.bool), paddle.finfo(dtype).min
            )

        expanded_4d_mask = expanded_attn_mask
        return expanded_4d_mask

    def to_4d_export(
        self,
        attention_mask_2d,
        query_length,
        dtype,
        key_value_length,
        use_parallel=False,
        parallel_step=3,
        is_export=False,
    ):
        input_shape = (attention_mask_2d.shape[0], query_length)

        expanded_attn_mask = self._expand_mask_export(
            attention_mask_2d, dtype, tgt_len=input_shape[-1]
        )
        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask

    def _expand_mask(self, mask, dtype, tgt_len=None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.shape
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = (
            mask[:, None, None, :].expand([bsz, 1, tgt_len, src_len]).cast(dtype)
        )

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill_(
            inverted_mask.cast(paddle.bool), paddle.finfo(dtype).min
        )

    def _expand_mask_export(self, mask, dtype, tgt_len=None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = paddle.shape(mask)
        expanded_mask = (
            mask[:, None, None, :].expand([bsz, 1, tgt_len, src_len]).cast(dtype)
        )
        paddle.jit.api.set_dynamic_shape(expanded_mask, [-1, -1, -1, -1])
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill_(
            inverted_mask.cast(paddle.bool), paddle.finfo(dtype).min
        )


def _prepare_4d_attention_mask(mask, dtype, tgt_len=None):
    return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _prepare_4d_causal_attention_mask(
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
    sliding_window=None,
    use_parallel=False,
    parallel_step=3,
    is_export=False,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`paddle.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `paddle.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`paddle.Tensor`):
            The embedded inputs as a paddle Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(
        is_causal=True, sliding_window=sliding_window
    )

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            key_value_length=key_value_length,
            dtype=inputs_embeds.dtype,
            use_parallel=use_parallel,
            parallel_step=parallel_step,
            is_export=is_export,
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape - invert it and fill with negative infinity
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


def _prepare_4d_causal_attention_mask_export(
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
    sliding_window=None,
    use_parallel=False,
    parallel_step=3,
    is_export=False,
):
    """
    Prepare a 4D causal attention mask for export.

    This function prepares a 4-dimensional causal attention mask, which is used to ensure that each position in the
    sequence can only attend to previous positions. It is specifically designed to handle scenarios where the model
    is being exported, potentially with additional options like sliding window or parallel processing.

    Args:
        attention_mask: The initial attention mask, typically used to avoid attending to padding tokens.
        input_shape: Shape of the input tensor, usually in the form (batch_size, sequence_length).
        inputs_embeds: Embeddings of the input sequence, used to derive certain dimensions if needed.
        past_key_values_length: Length of past key values, used in contexts like transformer decoders with caching.
        sliding_window: Optional parameter. If provided, specifies the size of a sliding window for local attention.
        use_parallel: Flag indicating whether to use parallel processing for attention computation.
        parallel_step: Number of steps to use in parallel processing, relevant if `use_parallel` is True.
        is_export: Flag indicating whether the attention mask is being prepared for model export.

    Returns:
        A 4D causal attention mask suitable for use in transformer models, ensuring correct causal masking.
    """
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
        use_parallel=use_parallel,
        parallel_step=parallel_step,
        is_export=is_export,
    )
    return attention_mask


class CustomMBartDecoder(MBartDecoder):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.d_model
        self.is_export = config.is_export
        self.config_decoder = config

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

        # retrieve input_ids and inputs_embeds
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

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            # 4d mask is passed through the layers
            if self.is_export:
                attention_mask = _prepare_4d_causal_attention_mask_export(
                    attention_mask,
                    input_shape,
                    inputs_embeds,
                    past_key_values_length,
                    use_parallel=self.config_decoder.use_parallel,
                    parallel_step=self.config_decoder.parallel_step,
                    is_export=self.is_export,
                )
            else:
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    input_shape,
                    inputs_embeds,
                    past_key_values_length,
                    use_parallel=self.config_decoder.use_parallel,
                    parallel_step=self.config_decoder.parallel_step,
                    is_export=self.is_export,
                )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self._use_flash_attention_2:
                encoder_attention_mask = (
                    encoder_attention_mask if 0 in encoder_attention_mask else None
                )
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
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
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
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

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.is_export:
            next_cache = next_decoder_cache
        else:
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


class CustomMBartForCausalLM(MBartForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # Modify the decoder within MBartDecoderWrapper
        self.model.decoder = CustomMBartDecoder(config)

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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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

        return CausalLMOutputWithCrossAttentions(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class PPFormulaNet_Head(UniMERNetHead):
    """
    PPFormulaNet_Head
    Args:
        max_new_tokens (int): Maximum number of new tokens to generate. Default is 1536.
        decoder_start_token_id (int): Start token ID for the decoder. Default is 0.
        temperature (float): Temperature parameter for controlling randomness in sampling. Default is 0.2.
        do_sample (bool): Flag to determine whether to use sampling for generation. Default is False.
        top_p (float): Top-p (nucleus) sampling parameter for controlling diversity. Default is 0.95.
        in_channels (int): Number of input channels for the model. Default is 1024.
        decoder_layers (int): Number of layers in the decoder. Default is 8.
        encoder_hidden_size (int): Size of the hidden layer in the encoder. Default is 1024.
        decoder_ffn_dim (int): Dimension of the feed-forward network in the decoder. Default is 4096.
        decoder_hidden_size (int): Size of the hidden layer in the decoder. Default is 1024.
        is_export (bool): Flag indicating whether the model is to be exported. Default is False.
        length_aware (bool): Flag to determine if the model should be aware of input sequence length. Default is True.
        use_parallel (bool): Flag to enable or disable parallel processing. Default is False.
        parallel_step (int): Number of steps to use in parallel processing. Default is 3.
    """

    def __init__(
        self,
        max_new_tokens=1536,
        decoder_start_token_id=0,
        temperature=0.2,
        do_sample=False,
        top_p=0.95,
        in_channels=1024,
        decoder_layers=8,
        encoder_hidden_size=1024,
        decoder_ffn_dim=4096,
        decoder_hidden_size=1024,
        is_export=False,
        length_aware=True,
        use_parallel=False,
        parallel_step=3,
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
            "max_position_embeddings": (
                max_new_tokens + parallel_step if use_parallel else max_new_tokens
            ),
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
            "use_parallel": use_parallel,
            "parallel_step": int(parallel_step),
            "is_export": is_export,
        }
        self.decoder_start_token_id = decoder_start_token_id
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.is_export = is_export
        self.max_seq_len = max_new_tokens
        self.config_decoder = MBartConfig(**mbart_config_dict)
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder = CustomMBartForCausalLM(self.config_decoder)
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
            "past_key_values": decoder_inputs["past_key_values"],
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
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], axis=-1
            )

        if not is_encoder_decoder:
            # update attention mask
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
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = paddle.concat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], 1)
                        ),
                    ],
                    dim=-1,
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

    def stopping_criteria_parallel(self, input_ids):
        parallel_step = self.config_decoder.parallel_step

        if self.is_export:
            is_done_list = []
            for i in range(parallel_step, 0, -1):
                cur_is_done = input_ids[:, -i] == paddle.to_tensor([self.eos_token_id])
                is_done_list.append(cur_is_done)
            is_done_list = paddle.to_tensor(is_done_list).transpose([1, 0])
            return is_done_list
        else:
            is_done = paddle.isin(
                input_ids[:, -parallel_step:],
                paddle.to_tensor([self.eos_token_id]).reshape([1, 1]),
            )
            return paddle.to_tensor(is_done)

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

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size,
        model_kwargs,
        decoder_start_token_id=None,
        bos_token_id=None,
    ):

        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
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
            use_parallel = self.config_decoder.use_parallel
            parallel_step = self.config_decoder.parallel_step

            if use_parallel:
                decoder_input_ids_start = (
                    paddle.ones(
                        (batch_size, parallel_step),
                        dtype=paddle.int64,
                    )
                    * decoder_start_token_id
                )
            else:
                decoder_input_ids_start = (
                    paddle.ones(
                        (batch_size, 1),
                        dtype=paddle.int64,
                    )
                    * decoder_start_token_id
                )
        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
        elif (
            self.config.model_type == "vision-encoder-decoder"
            and "donut" in self.name_or_path.lower()
        ):
            pass
        elif self.config.model_type in ["whisper"]:
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
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

    @paddle.no_grad()
    def generate_export(
        self,
        encoder_outputs,
        model_kwargs,
    ):
        use_parallel = self.config_decoder.use_parallel
        parallel_step = self.config_decoder.parallel_step
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
        if not use_parallel:
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
        if use_parallel:
            unfinished_sequences = paddle.ones(
                [batch_size, parallel_step], dtype=paddle.int64
            )
            parallel_length = math.ceil(self.max_seq_len // parallel_step)
        else:
            unfinished_sequences = paddle.ones(batch_size, dtype=paddle.int64)
            parallel_length = self.max_seq_len

        i_idx = paddle.full([], 0)
        past_key_values = []
        decoder_attention_heads = self.config_decoder.decoder_attention_heads
        decoder_attention_heads_dim = int(
            self.config_decoder.d_model / decoder_attention_heads
        )
        for i in range(self.config_decoder.decoder_layers):
            init_arr = paddle.zeros(
                [batch_size, decoder_attention_heads, 0, decoder_attention_heads_dim]
            )
            paddle.jit.api.set_dynamic_shape(init_arr, [-1, -1, -1, -1])
            cache = (init_arr, init_arr, init_arr, init_arr)
            past_key_values.append(cache)

        while i_idx < paddle.to_tensor(parallel_length):

            model_inputs = self.prepare_inputs_for_generation_export(
                past_key_values=past_key_values, **model_kwargs
            )
            decoder_attention_mask = paddle.ones(paddle.shape(input_ids))
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

            if use_parallel:
                next_token_logits = outputs.logits[:, -parallel_step:, :]
            else:
                next_token_logits = outputs.logits[:, -1, :]
            next_tokens_scores = self.logits_processor(input_ids, next_token_logits)
            next_tokens = paddle.argmax(next_tokens_scores, axis=-1)

            if eos_token_id is not None:
                # False
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )
            if use_parallel:
                input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
                decoder_input_ids = next_tokens
            else:
                input_ids = paddle.concat(
                    [input_ids, next_tokens.unsqueeze(1)], axis=-1
                )
                decoder_input_ids = next_tokens.unsqueeze(1)

            past_length = past_key_values[0][0].shape[2]

            past_key_values = outputs.past_key_values
            cache_position = cache_position[-1:] + 1
            if use_parallel:
                unfinished_sequences = (
                    unfinished_sequences
                    & ~self.stopping_criteria_parallel(input_ids).cast(paddle.int64)
                )
            else:
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
            # break

        return input_ids

    @paddle.no_grad()
    def generate(
        self,
        encoder_outputs,
        model_kwargs,
    ):
        """
        Generate sequences from the model without computing gradients.

        This method is used to generate sequences from the model based on the given encoder outputs.
        It does not compute gradients, making it suitable for inference.

        Args:
            encoder_outputs: The outputs from the encoder, typically including hidden states necessary for generation.
            model_kwargs: Additional keyword arguments that may include parameters such as maximum length,
                        temperature, top-k/top-p sampling parameters, and other generation-specific settings.

        Returns:
            Generated sequences based on the encoder outputs and specified generation parameters.
        """
        use_parallel = self.config_decoder.use_parallel
        parallel_step = self.config_decoder.parallel_step
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

        decoder_input_ids = input_ids
        model_kwargs["key use_cache"] = True
        batch_size, cur_len = input_ids.shape

        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        model_kwargs["cache_position"] = paddle.arange(cur_len)
        pad_token_id = self.pad_token_id
        eos_token_id = [self.eos_token_id]
        eos_token = self.eos_token_id
        if use_parallel:
            unfinished_sequences = paddle.ones(
                [batch_size, parallel_step], dtype=paddle.int64
            )
            parallel_length = math.ceil(self.max_seq_len // parallel_step)
        else:
            unfinished_sequences = paddle.ones(batch_size, dtype=paddle.int64)
            parallel_length = self.max_seq_len
        past_key_values = []

        for idx in range(parallel_length):

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self.generate_single_iter(
                **model_inputs,
                encoder_outputs=encoder_outputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            if use_parallel:
                next_token_logits = outputs.logits[:, :, :]
            else:
                next_token_logits = outputs.logits[:, -1, :]

            next_tokens_scores = self.logits_processor(input_ids, next_token_logits)
            next_tokens = paddle.argmax(next_tokens_scores, axis=-1)
            if eos_token_id is not None:
                # False
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )
            if use_parallel:
                input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
            else:
                input_ids = paddle.concat([input_ids, next_tokens[:, None]], axis=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config_decoder.is_encoder_decoder,
            )
            if use_parallel:
                unfinished_sequences = (
                    unfinished_sequences
                    & ~self.stopping_criteria_parallel(input_ids).cast(paddle.int64)
                )
            else:
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
        Forward pass for training the model.

        Args:
            encoder_outputs: The outputs from the encoder, typically including hidden states.
            decoder_input_ids: Input IDs for the decoder.
            decoder_attention_mask: Attention mask for the decoder inputs to avoid attending to padding tokens.
            past_key_values: Previously computed key and value states for the decoder, used for fast generation.
            decoder_inputs_embeds: Optional embeddings for decoder inputs, used instead of decoder_input_ids if provided.
            labels: Labels for computing the training loss.
            use_cache: Whether to use a cache of past key values for faster generation.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states of all layers.
            return_dict: Whether to return the output as a dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            Depending on the `return_dict` flag, returns either a dictionary of model outputs or a tuple.
        """
        if self.config_decoder.use_parallel:
            batch = decoder_input_ids.shape[0]
            add_sos_token = self.config_decoder.parallel_step - 1
            start_token = paddle.zeros([batch, add_sos_token]).cast(paddle.int64)
            start_mask = paddle.ones([batch, add_sos_token]).cast(paddle.int64)
            decoder_input_ids = paddle.concat([start_token, decoder_input_ids], axis=1)
            decoder_attention_mask = paddle.concat(
                [start_mask, decoder_attention_mask], axis=1
            )

        labels = decoder_input_ids * 1
        labels = labels.masked_fill_(labels == self.pad_token_id, -100)
        if self.config_decoder.use_parallel:
            input_decoder_input_ids = decoder_input_ids[
                :, : -self.config_decoder.parallel_step
            ]
            input_decoder_attention_mask = decoder_attention_mask[
                :, : -self.config_decoder.parallel_step
            ]
        else:
            input_decoder_input_ids = decoder_input_ids[:, :-1]
            input_decoder_attention_mask = decoder_attention_mask[:, :-1]

        encoder_hidden_states = encoder_outputs[0]
        kwargs_decoder = {}
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

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
        return logits, labels

    # forward for export
    def forward(self, inputs, targets=None):
        self.is_export = False if self.training else True
        if not self.training:
            encoder_outputs = inputs
            model_kwargs = {
                "output_attentions": False,
                "output_hidden_states": False,
                "use_cache": True,
            }
            if self.is_export:
                word_pred = self.generate_export(encoder_outputs, model_kwargs)
            else:
                word_pred = self.generate(encoder_outputs, model_kwargs)

            return word_pred
        encoder_outputs, tgt_seq, mask = inputs
        logits, masked_labels = self.forwad_train(encoder_outputs, tgt_seq, mask)

        return logits, masked_labels
