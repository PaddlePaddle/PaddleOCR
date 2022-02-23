# encoding=utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The Facebook, Inc. and The HuggingFace Inc. team.
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

import numpy as np
import math
import paddle
import paddle.nn as nn
import paddle.tensor as tensor
from paddle.nn import Embedding
from .. import PretrainedModel, register_base_model
from paddle.nn.layer.transformer import _convert_attention_mask

__all__ = [
    'BlenderbotSmallModel', 'BlenderbotSmallPretrainedModel',
    'BlenderbotSmallEncoder', 'BlenderbotSmallDecoder',
    'BlenderbotSmallForConditionalGeneration', 'BlenderbotSmallForCausalLM'
]


# Copied from paddlenlp.transformers.bart.modeling.shift_tokens_right
def shift_tokens_right(input_ids: tensor, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = paddle.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    return shifted_input_ids


class BlenderbotSmallLearnedPositionalEmbedding(Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.

    Please should refer to the superclass for more information regarding methods and arguments.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, input_ids_shape, past_key_values_length=0):
        """
        Generate positional embeddings up based on input_ids_shape.
        Args:
            input_ids_shape (`tuple`): expected to be [batch_size, sequence_length].
            past_key_values_length (`int`, optional): The length of past_key_value,
            which is used only when the ``use_cache=True`` during prediction generating.

        Returns:
            (Tensor): The generated positional embedding.
        """
        bsz, seq_len = input_ids_shape[:2]
        positions = paddle.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype="int64")
        return super().forward(positions)


class BlenderbotSmallPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained BlenderbotSmall models. It provides BlenderbotSmall related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "blenderbot_small-90M": {
            "vocab_size": 54944,
            "bos_token_id": 1,
            "pad_token_id": 0,
            "eos_token_id": 2,
            "decoder_start_token_id": 1,
            "d_model": 512,
            "num_encoder_layers": 8,
            "num_decoder_layers": 8,
            "encoder_attention_heads": 16,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": 2048,
            "encoder_ffn_dim": 2048,
            "dropout": 0.1,
            "activation_function": "gelu",
            "init_std": 0.02,
            "max_position_embeddings": 512,
            "attention_dropout": 0.0,
            "activation_dropout": 0.0,
            "scale_embedding": True,
            "normalize_before": False,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "blenderbot_small-90M":
            "https://bj.bcebos.com/paddlenlp/models/transformers/blenderbot_small/blenderbot_small-90M.pdparams",
        }
    }
    base_model_prefix = "blenderbot_small"

    def init_weights(self, layer):
        """ Initialization hook """
        if paddle.get_default_dtype() not in ['float32', 'float64']:
            # gaussian/standard_normal/randn/normal only supports [float32, float64]
            return
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.init_std if hasattr(self, "init_std") else
                        self.blenderbot_small.config["init_std"],
                        shape=layer.weight.shape))


class BlenderbotSmallDecoderLayer(nn.TransformerDecoderLayer):
    """
    Construct decoder layer for BlenderbotSmallDecoder.
    Please refer to :class:`~paddlenlp.nn.TransformerDecoderLayer` for more details.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="gelu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=True,
                 weight_attr=None,
                 bias_attr=None):
        super(BlenderbotSmallDecoderLayer, self).__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            attn_dropout=attn_dropout,
            act_dropout=act_dropout,
            normalize_before=normalize_before,
            weight_attr=weight_attr,
            bias_attr=bias_attr)

    def forward(self,
                tgt,
                memory=None,
                tgt_mask=None,
                memory_mask=None,
                cache=None):
        """
        Please refer to  :class:`~paddlenlp.nn.TransformerDecoderLayer`
        for more information regarding arguments.
        """
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if cache is None:
            tgt = self.self_attn(
                query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask, cache=None)
        else:
            tgt, incremental_cache = self.self_attn(
                query=tgt,
                key=tgt,
                value=tgt,
                attn_mask=tgt_mask,
                cache=cache[0])
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        # Cross-attention will not be applied for BlenderbotSmallForCausalLM
        if memory is not None:
            residual = tgt
            if self.normalize_before:
                tgt = self.norm2(tgt)
            memory_mask = _convert_attention_mask(memory_mask, memory.dtype)
            if cache is None:
                tgt = self.cross_attn(
                    query=tgt,
                    key=memory,
                    value=memory,
                    attn_mask=memory_mask,
                    cache=None)
            else:
                tgt, static_cache = self.cross_attn(
                    query=tgt,
                    key=memory,
                    value=memory,
                    attn_mask=memory_mask,
                    cache=cache[1])
            tgt = residual + self.dropout2(tgt)
            if not self.normalize_before:
                tgt = self.norm2(tgt)
        else:
            static_cache = cache[1] if cache is not None else None

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt if cache is None else (tgt, (incremental_cache,
                                                static_cache))


class TransformerDecoder(nn.TransformerDecoder):
    """
    Construct Transformer decoder for BlenderbotSmallDecoder.
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__(
            decoder_layer=decoder_layer, num_layers=num_layers, norm=norm)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        """
        Please refer to  :class:`~paddlenlp.nn.TransformerDecoder`
        for more information regarding arguments and methods.
        """

        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        if memory is not None:
            memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        output = tgt
        new_caches = []
        for i, mod in enumerate(self.layers):
            if cache is None:
                output = mod(output,
                             memory,
                             tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             cache=None)
            else:
                output, new_cache = mod(output,
                                        memory,
                                        tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        cache=cache[i])
                new_caches.append(new_cache)

        if self.norm is not None:
            output = self.norm(output)

        return output if cache is None else (output, new_caches)


class BlenderbotSmallEncoder(BlenderbotSmallPretrainedModel):
    """
    The encoder of BlenderbotSmall Model.
    Please refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` or
    :class:`~paddlenlp.transformers.Blenderbot.BlenderbotSmallModel` for more details
    regarding methods and arguments.
    """

    def __init__(self,
                 vocab_size,
                 embed_tokens=None,
                 pad_token_id=0,
                 d_model=512,
                 num_encoder_layers=6,
                 encoder_attention_heads=12,
                 encoder_ffn_dim=2048,
                 dropout=0.1,
                 activation_function='gelu',
                 attention_dropout=0.0,
                 activation_dropout=0.0,
                 max_position_embeddings=1024,
                 init_std=0.02,
                 scale_embedding=True,
                 normalize_before=False):
        super().__init__()
        self.init_std = init_std
        self.pad_token_id = pad_token_id
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=d_model,
                padding_idx=pad_token_id)
        self.encoder_embed_positions = BlenderbotSmallLearnedPositionalEmbedding(
            num_embeddings=max_position_embeddings, embedding_dim=d_model)
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.encoder_dropout = nn.Dropout(dropout)
        self.encoder_layernorm_embedding = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=encoder_attention_heads,
            dim_feedforward=encoder_ffn_dim,
            dropout=dropout,
            activation=activation_function,
            attn_dropout=attention_dropout,
            act_dropout=activation_dropout,
            normalize_before=normalize_before)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers)
        self.apply(self.init_weights)

    def forward(self, input_ids=None, attention_mask=None):
        """
        Returns:
            Tensor: The last hidden-states at the last layer of the encoder.
            It's data type should be `float` and has a shape of `(batch_size, seq_lens, hidden_size)`.
            ``seq_lens`` corresponds to the length of input sequence.
        """
        if input_ids is None:
            raise ValueError("Input_ids cannot be None.")
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        inputs_embed_pos = self.encoder_embed_positions(input_ids.shape)
        hidden_states = inputs_embeds + inputs_embed_pos
        hidden_states = self.encoder_layernorm_embedding(hidden_states)
        encoder_input = self.encoder_dropout(hidden_states)

        if attention_mask is None:
            attention_mask = paddle.cast(
                input_ids == self.pad_token_id,
                dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
            attention_mask.stop_gradient = True

        encoder_output = self.encoder(encoder_input, src_mask=attention_mask)
        return encoder_output


class BlenderbotSmallDecoder(BlenderbotSmallPretrainedModel):
    """
    The decoder of BlenderbotSmall Model.
    Please refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` and
    :class:`~paddlenlp.transformers.Blenderbot.BlenderbotModel` for more information
    regarding methods and arguments.
    """

    def __init__(self,
                 vocab_size,
                 embed_tokens=None,
                 pad_token_id=1,
                 d_model=768,
                 num_decoder_layers=6,
                 decoder_attention_heads=12,
                 decoder_ffn_dim=3072,
                 dropout=0.1,
                 activation_function='gelu',
                 attention_dropout=0.1,
                 activation_dropout=0.1,
                 max_position_embeddings=1024,
                 init_std=0.02,
                 scale_embedding=True,
                 normalize_before=False):
        super().__init__()
        self.init_std = init_std
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=d_model,
                padding_idx=pad_token_id)

        self.decoder_embed_positions = BlenderbotSmallLearnedPositionalEmbedding(
            num_embeddings=max_position_embeddings, embedding_dim=d_model)
        self.decoder_dropout = nn.Dropout(dropout)
        self.decoder_layernorm_embedding = nn.LayerNorm(
            normalized_shape=d_model)
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0

        decoder_layer = BlenderbotSmallDecoderLayer(
            d_model=d_model,
            nhead=decoder_attention_heads,
            dim_feedforward=decoder_ffn_dim,
            dropout=dropout,
            activation=activation_function,
            attn_dropout=attention_dropout,
            act_dropout=activation_dropout,
            normalize_before=normalize_before)
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_decoder_layers)
        self.apply(self.init_weights)

    def forward(self,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_output=None,
                memory_mask=None,
                use_cache=False,
                cache=None):
        """
        Please refer to :class:`~paddlenlp.transformers.Blenderbot.BlenderbotModel` for more
        information regarding the arguments.
        """
        if decoder_input_ids is None:
            raise ValueError("Decoder_input_ids cannot be None.")
        if decoder_attention_mask is None:
            decoder_length = paddle.shape(decoder_input_ids)[-1]
            decoder_attention_mask = paddle.tensor.triu(
                (paddle.full(
                    (decoder_length, decoder_length),
                    -np.inf,
                    dtype=paddle.get_default_dtype())),
                1)
        decoder_inputs_embeds = self.embed_tokens(
            decoder_input_ids) * self.embed_scale
        # cache[num_layer][0] is an instance of `MultiHeadAttention.Cache` containing
        # tensor k and v with shape of `[batch_size, num_heads, len_seq, embed_dim // num_heads]`
        # ``len_seq`` refer to the length of ``decoder_input_ids``
        # Refer to paddle.nn.MultiHeadAttention.gen_cache for more details regarding cache.
        past_key_values_length = cache[0][0].k.shape[
            2] if cache is not None else 0

        decoder_inputs_embed_pos = self.decoder_embed_positions(
            input_ids_shape=decoder_input_ids.shape,
            past_key_values_length=past_key_values_length)

        # Different from BLenderbot, BlenderbotSmall Apply layer norm on decoder_inputs_embeds
        decoder_inputs_embeds = self.decoder_layernorm_embedding(
            decoder_inputs_embeds)

        hidden_states = decoder_inputs_embeds + decoder_inputs_embed_pos
        decoder_input = self.decoder_dropout(hidden_states)

        decoder_output = self.decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=decoder_attention_mask,
            memory_mask=memory_mask,
            cache=cache)
        return decoder_output


@register_base_model
class BlenderbotSmallModel(BlenderbotSmallPretrainedModel):
    r"""
     Construct a bare BlenderbotSmall Model.

     This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
     Check the superclass documentation for the generic methods and the library implements for all its model.

     This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
     /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
     and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
         vocab_size (`int`):
             Vocabulary size of the BlenderbotSmall model.
         bos_token_id (`int`, optional):
            The id for begging of sentences token. Defaults to ``1``.
         pad_token_id (`int`, optional):
            The id for padding token. Defaults to ``0``.
         eos_token_id (`int`, optional):
            The id for end of sentence token. Defaults to ``2``.
         decoder_start_token_id (`int`, optional):
            The id indicating the start of decoding sentence. Defaults to ``1``.
         d_model (`int`, optional):
            Dimensionality of the layers and the pooler layer. Defaults to ``512``.
         num_encoder_layers (`int`, optional):
            Number of Transformer encoder layers for BlenderbotSmallEncoder. Defaults to ``8``.
         num_decoder_layers (`int`, optional):
            Number of Transformer decoder layers for BlenderbotSmallDecoder. Defaults to ``8``.
         encoder_attention_heads (`int`, optional):
            Number of attention heads for each Transformer encoder layer in BlenderbotSmallEncoder.
            Defaults to ``16``.
         decoder_attention_heads (`int`, optional):
            Number of attention heads for each Transformer decoder layer in BlenderbotSmallDecoder.
            Defaults to ``16``.
         encoder_ffn_dim (`int`, optional):
            Dimensionality of the feed-forward layer for each Transformer encoder layer in
            BlenderbotSmallEncoder. Defaults to ``2048``.
         decoder_ffn_dim (`int`, optional):
            Dimensionality of the feed-forward layer for each Transformer dncoder layer in
            BlenderbotSmallDncoder. Defaults to ``2048``.
         dropout (`float`, optional):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            Defaults to ``0.1``.
         activation_function (`str`, optional):
            The non-linear activation function (function or string) in the encoder and pooler.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to ``"gelu"``.
         attention_dropout (`float`, optional):
            The dropout ratio for the attention probabilities.
            Defaults to ``0.0``.
         activation_dropout (`float`, optional):
            The dropout ratio for activations inside the fully connected layer.
         max_position_embeddings (`int`, optional):,
            The max position index of an input sequence. Defaults to ``512``.
         init_std (`float`, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Defaults to ``0.02``.
         scale_embedding (`bool`, optional):
            Indicate whether to scale embeddings by diving by sqrt(d_model). Defaults to ``True``.
         normalize_before (bool, optional):
            Indicate whether to put layer normalization into preprocessing of MHA and FFN sub-layers.
            If True, pre-process is layer normalization and post-precess includes dropout,
            residual connection. Otherwise, no pre-process and post-precess includes dropout,
            residual connection, layer normalization. Defaults to ``False``.
    """

    def __init__(self,
                 vocab_size,
                 bos_token_id=1,
                 pad_token_id=0,
                 eos_token_id=2,
                 decoder_start_token_id=1,
                 d_model=512,
                 num_encoder_layers=8,
                 num_decoder_layers=8,
                 encoder_attention_heads=16,
                 decoder_attention_heads=16,
                 encoder_ffn_dim=2048,
                 decoder_ffn_dim=2048,
                 dropout=0.1,
                 activation_function='gelu',
                 attention_dropout=0.0,
                 activation_dropout=0.0,
                 max_position_embeddings=512,
                 init_std=0.02,
                 scale_embedding=True,
                 normalize_before=False):
        super().__init__()
        self.init_std = init_std
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.shared = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_token_id)
        self.encoder = BlenderbotSmallEncoder(
            vocab_size=vocab_size,
            embed_tokens=self.shared,
            pad_token_id=pad_token_id,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            dropout=dropout,
            activation_function=activation_function,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            max_position_embeddings=max_position_embeddings,
            init_std=init_std,
            scale_embedding=scale_embedding,
            normalize_before=normalize_before)

        self.decoder = BlenderbotSmallDecoder(
            vocab_size=vocab_size,
            embed_tokens=self.shared,
            pad_token_id=pad_token_id,
            d_model=d_model,
            num_decoder_layers=num_decoder_layers,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            dropout=dropout,
            activation_function=activation_function,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            max_position_embeddings=max_position_embeddings,
            init_std=init_std,
            scale_embedding=scale_embedding,
            normalize_before=normalize_before)
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_output=None,
                use_cache=False,
                cache=None,
                **kwargs):
        r"""
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].

            attention_mask (Tensor, optional):
                Mask to indicate whether to perform attention on each input token or not.
                The values should be either 0 or 1. The attention scores will be set
                to **-infinity** for any positions in the mask that are **0**, and will be
                **unchanged** for positions that are **1**.

                - **1** for tokens that are **not masked**,
                - **0** for tokens that are **masked**.

                It's data type should be `float32` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.

            decoder_input_ids (Tensor, optional):
                If not provided, ``decoder_input_ids`` will be automatically generated based
                on ``decoder_start_token_id`` and ``input_ids``.

            decoder_attention_mask (Tensor, optional):
                If not provided, the default ``decoder_attention_mask`` will be a tensor with
                upper triangular part being ``-np.inf``. the shape will be ``(decoder_length, decoder_length)``

            encoder_output (Tensor, optional):
                The output of encoder. If not provided, a new ``encoder_output`` will be generated
                from BlenderbotEncoder. Defaults to ``None``.

            use_cache (bool, optional):
                Indicates whether to use cache to speed up decoding. Defaults to ``False``

            cache (list, optional): It is a list, and each element in the list
                is a tuple( :code:`(incremental_cache, static_cache)` ). See
                `TransformerDecoder.gen_cache` for more details. It is only
                used for inference and should be None for training. Default None.
        Returns:
            Tensor|tuple:
                If ``use_cache=False``, the return will be the last hidden state of decoder with shape
                of [batch_size, seq_lens, hidden_size]. ``seq_lens`` corresponds to the length of input sequence.
                Otherwise, the return will be a tuple of ``(decoder_output, cache)``. Please refer to
                class :class:`paddle.nn.TransformerDecoder` for more information regarding ``cache``.

        Example:
            .. code-block::

            import paddle
            from paddlenlp.transformers import BlenderbotSmallTokenizer, BlenderbotSmallModel

            # "blenderbot_small-90M" is pretrained weight of BlenderbotSmallForConditionalGeneration,
            # Therefore some weight of additional layers in BlenderbotSmallForConditionalGeneration
            # might not be loaded and used.
            pretrained_model_name = "blenderbot_small-90M"
            tokenizer = BlenderbotSmallTokenizer.from_pretrained(pretrained_model_name)
            model = BlenderbotSmallModel.from_pretrained(pretrained_model_name)

            sample_text = "My friends are cool but they eat too many carbs."
            inputs = tokenizer(sample_text, return_attention_mask=True, return_token_type_ids=False)
            inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
            decoder_output = model(**inputs)
        """
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids=input_ids,
                decoder_start_token_id=self.decoder_start_token_id)
        if encoder_output is None:
            encoder_output = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask)
        # initialize cache based on encoder output for decoding at 1st time step.
        if use_cache:
            if cache is None:
                cache = self.decoder.decoder.gen_cache(encoder_output)
        else:
            cache = None

        if attention_mask is None:
            assert input_ids is not None, "input_ids should be " \
                                          "specified when generating attention_mask"
            memory_mask = paddle.cast(
                input_ids == self.pad_token_id,
                dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
            memory_mask.stop_gradient = True
        else:
            memory_mask = attention_mask

        decoder_output = self.decoder(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_output=encoder_output,
            memory_mask=memory_mask,
            use_cache=use_cache,
            cache=cache)
        return decoder_output

    def get_encoder(self):
        """
        This method is required for model with encoder-decoder architecture.
        """
        return self.encoder


class BlenderbotSmallForConditionalGeneration(BlenderbotSmallPretrainedModel):
    """
    Please refer to :class:`~paddlenlp.transformers.Blenderbot.BlenderbotModel` for more
    information regarding arguments.
    Return:
        Tensor|tuple: If ``use_cache=False``, the return will be a tensor with shape of
            [batch_size, seq_lens, hidden_size]. Otherwise, the return will be a tuple
            of ``(decoder_output, cache)``.
    Example:
        .. code-block::

            import paddle
            from paddlenlp.transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration

            pretrained_model_name = "blenderbot_small-90M"
            tokenizer = BlenderbotSmallTokenizer.from_pretrained(pretrained_model_name)
            model = BlenderbotSmallForConditionalGeneration.from_pretrained(pretrained_model_name)

            sample_text = "My friends are cool but they eat too many carbs."
            inputs = tokenizer(sample_text, return_attention_mask=True, return_token_type_ids=False)
            inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}
            result_ids, score = model.generate(input_ids=inputs['input_ids'],
                                               max_length=60,
                                               min_length=20,
                                               decode_strategy='beam_search',
                                               num_beams=10,
                                               length_penalty=0.65
                                               )
            for sequence_ids in result_ids.numpy().tolist():
                print("User:\t", sample_text)
                print("bot:\t", tokenizer.convert_ids_to_string(sequence_ids))
    """

    def __init__(self, blenderbot_small):
        super(BlenderbotSmallForConditionalGeneration, self).__init__()
        self.eos_token_id = blenderbot_small.eos_token_id
        self.bos_token_id = blenderbot_small.bos_token_id
        self.pad_token_id = blenderbot_small.pad_token_id
        self.blenderbot_small = blenderbot_small
        self.lm_head_weight = self.create_parameter(
            shape=[
                self.blenderbot_small.config['vocab_size'],
                self.blenderbot_small.config['d_model']
            ],
            dtype=self.blenderbot_small.shared.weight.dtype,
            is_bias=False)
        self.register_buffer(
            "final_logits_bias",
            paddle.zeros(
                (1, self.blenderbot_small.config['vocab_size']),
                dtype=paddle.get_default_dtype()))
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_output=None,
                use_cache=False,
                cache=None):
        decoder_outputs = self.blenderbot_small(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_output=encoder_output,
            use_cache=use_cache,
            cache=cache)

        lm_logits = paddle.tensor.matmul(
            decoder_outputs[0] if use_cache else decoder_outputs,
            self.lm_head_weight,
            transpose_y=True) + self.final_logits_bias
        if use_cache:
            cache = decoder_outputs[1]
            return lm_logits, cache
        return lm_logits

    def prepare_inputs_for_generation(self,
                                      decoder_input_ids,
                                      attention_mask=None,
                                      encoder_output=None,
                                      use_cache=True,
                                      cache=None,
                                      **kwargs):

        if encoder_output is not None:
            expand_size = int(decoder_input_ids.shape[0] /
                              encoder_output.shape[0])
            if expand_size > 1:
                index = paddle.tile(
                    paddle.arange(encoder_output.shape[0]).unsqueeze(-1),
                    [1, expand_size]).reshape([-1])
                encoder_output = paddle.index_select(encoder_output, index)

        if use_cache and cache is None:
            if encoder_output is None:
                raise ValueError(
                    "Encoder output can not be none if `use_cache` is True")
            cache = self.decoder.decoder.gen_cache(memory=encoder_output)

        if cache is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {
            "input_ids":
            None,  # during prediction, Encoder_output is provided, do not need input_ids.
            "decoder_input_ids": decoder_input_ids,
            "encoder_output": encoder_output,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache
        }

    def get_encoder(self):
        """
        This method is required for model with encoder-decoder architecture.
        """
        return self.encoder

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


class BlenderbotSmallForCausalLM(BlenderbotSmallPretrainedModel):
    """
    Constructs BLenderbotSmall For Causal Language Model. This model is equivalent to the
    blenderbotSmall decoder without cross-attention.
    """

    def __init__(self, blenderbot_small):
        super().__init__()
        self.blenderbot_small = blenderbot_small
        self.decoder = blenderbot_small.decoder

        self.lm_head_weight = self.create_parameter(
            shape=[
                blenderbot_small.config['vocab_size'],
                blenderbot_small.config['d_model']
            ],
            dtype=blenderbot_small.shared.weight.dtype,
            is_bias=False)
        self.register_buffer(
            "final_logits_bias",
            paddle.zeros(
                (1, blenderbot_small.config['vocab_size']),
                dtype=paddle.get_default_dtype()))
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                use_cache=False,
                cache=None,
                **kwargs):
        """
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].

            attention_mask (Tensor, optional):
                Mask to indicate whether to perform attention on each input token or not.
                The values should be either 0 or 1. The attention scores will be set
                to **-infinity** for any positions in the mask that are **0**, and will be
                **unchanged** for positions that are **1**.

                - **1** for tokens that are **not masked**,
                - **0** for tokens that are **masked**.

                It's data type should be `float32` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.

            use_cache (bool, optional):
                Indicates whether to use cache to speed up decoding. Defaults to ``False``

            cache (list, optional): It is a list, and each element in the list
                is a tuple( :code:`(incremental_cache, static_cache)` ). See
                `paddle.nn.TransformerDecoder.gen_cache` for more details. It is only
                used for inference and should be None for training. Default None.
        Return:
            Tensor|tuple: If ``use_cache=False``, the return will be a tensor with shape of
                [batch_size, seq_lens, hidden_size]. Otherwise, the return will be a tuple
                of ``(lm_logits, cache)``.
        Example:
            .. code-block::

            import paddle
            from paddlenlp.transformers import BlenderbotSmallTokenizer, BlenderbotSmallForCausalLM
            use_cache = False
            text = "My friends are cool but they eat too many carbs."
            model_name = "blenderbot_small-90M"
            tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name)
            model = BlenderbotSmallForCausalLM.from_pretrained(model_name)
            model.eval()
            inputs = tokenizer(text, return_attention_mask=True, return_token_type_ids=False)
            inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}

            with paddle.no_grad():
                outputs = model(**inputs, use_cache=use_cache)
                # outputs is a tuple of (lm_logits, cache) if ``use_cache=True``.

        """
        if use_cache and cache is None:
            # Generating incremental cache. A random tensor with shape of
            # (batch_size, len_seq, hidden_size) is passed for memory argument.
            # since the `static_cache` will not be used in BlenderbotSmallForCausalLM
            batch_size, len_seq = input_ids.shape
            cache = self.decoder.decoder.gen_cache(memory=paddle.zeros(
                (batch_size, len_seq, self.blenderbot_small.config['d_model'])))
        decoder_outputs = self.decoder(
            decoder_input_ids=input_ids,
            encoder_output=None,
            memory_mask=None,
            use_cache=use_cache,
            cache=cache)

        lm_logits = paddle.tensor.matmul(
            decoder_outputs[0] if use_cache else decoder_outputs,
            self.lm_head_weight,
            transpose_y=True) + self.final_logits_bias

        if use_cache:
            cache = decoder_outputs[1]
            return lm_logits, cache
        return lm_logits

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      attention_mask=None,
                                      use_cache=True,
                                      cache=None,
                                      **kwargs):
        """
        Prepare inputs for decoder to generate sentences.
        Return:
            dict: A dictionary containing necessary inputs for generating next token.
        """
        if cache is not None:
            input_ids = input_ids[:, -1:].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache
        }
