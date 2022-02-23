# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import tensor
from paddle.nn import Layer

from .. import PretrainedModel, register_base_model

__all__ = [
    "RoFormerModel",
    "RoFormerPretrainedModel",
    "RoFormerForPretraining",
    "RoFormerPretrainingCriterion",
    "RoFormerPretrainingHeads",
    "RoFormerForSequenceClassification",
    "RoFormerForTokenClassification",
    "RoFormerForQuestionAnswering",
]

dtype_float = paddle.get_default_dtype()


def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = attn_mask.dtype
        if attn_mask_dtype in [
                paddle.bool, paddle.int8, paddle.int16, paddle.int32,
                paddle.int64
        ]:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


class MultiHeadAttentionWithRotary(Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 rotary_value=False):
        super(MultiHeadAttentionWithRotary, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.need_weights = need_weights
        self.rotary_value = rotary_value

        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(self.kdim, embed_dim)
        self.v_proj = nn.Linear(self.vdim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def positional_embedding(self, inputs):
        seq_len = inputs.shape[1]
        pos_seq = paddle.arange(0, seq_len, dtype=dtype_float)
        indices = paddle.arange(0, self.head_dim, 2, dtype=dtype_float)
        indices = 1 / 10000**(indices / self.head_dim)
        sinusoid_inp = paddle.einsum("i,d->id", pos_seq, indices)
        pos_emb = paddle.concat(
            [paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)], axis=-1)
        pos_emb = paddle.reshape(pos_emb, (1, 1, seq_len, self.head_dim))
        pos_emb.stop_gradient = True
        return pos_emb

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos,
                                         query_layer,
                                         key_layer,
                                         value_layer=None):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = paddle.chunk(sinusoidal_pos, 2, axis=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = paddle.reshape(
            paddle.stack(
                [sin, sin], axis=-1), sinusoidal_pos.shape)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = paddle.reshape(
            paddle.stack(
                [cos, cos], axis=-1), sinusoidal_pos.shape)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = paddle.reshape(
            paddle.stack(
                [-query_layer[:, :, :, 1::2], query_layer[:, :, :, 0::2]],
                axis=-1),
            query_layer.shape, )
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = paddle.reshape(
            paddle.stack(
                [-key_layer[:, :, :, 1::2], key_layer[:, :, :, 0::2]], axis=-1),
            key_layer.shape, )
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = paddle.reshape(
                paddle.stack(
                    [-value_layer[:, :, :, 1::2], value_layer[:, :, :, 0::2]],
                    axis=-1),
                value_layer.shape, )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])

        sinusoidal_pos = self.positional_embedding(query)
        if self.rotary_value:
            q, k, v = self.apply_rotary_position_embeddings(sinusoidal_pos, q,
                                                            k, v)
        else:
            q, k = self.apply_rotary_position_embeddings(sinusoidal_pos, q, k)

        product = tensor.matmul(x=q, y=k, transpose_y=True) * self.scale
        if attn_mask is not None:
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask

        weights = F.softmax(product)
        weights = self.dropout(weights)
        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerEncoderLayerWithRotary(nn.TransformerEncoderLayer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 rotary_value=False,
                 **kwargs):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout=dropout,
            activation=activation,
            attn_dropout=attn_dropout,
            act_dropout=act_dropout,
            normalize_before=normalize_before)
        self.self_attn = MultiHeadAttentionWithRotary(
            d_model, nhead, dropout=attn_dropout, rotary_value=rotary_value)
        self._config.update({"rotary_value": rotary_value})


class RoFormerEmbeddings(Layer):
    """
    Include embeddings from word and token_type embeddings
    """

    def __init__(
            self,
            vocab_size,
            embedding_size=768,
            hidden_dropout_prob=0.1,
            type_vocab_size=2, ):
        super(RoFormerEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size,
                                                  embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):

        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RoFormerPooler(Layer):
    """ """

    def __init__(self, hidden_size, pool_act="tanh"):
        super(RoFormerPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.pool_act = pool_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == "tanh":
            pooled_output = self.activation(pooled_output)
        return pooled_output


class RoFormerPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained RoFormer models. It provides RoFormer related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "roformer-chinese-small": {
            "vocab_size": 50000,
            "embedding_size": 384,
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 6,
            "intermediate_size": 1536,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "rotary_value": False,
        },
        "roformer-chinese-base": {
            "vocab_size": 50000,
            "embedding_size": 768,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1536,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "rotary_value": False,
        },
        "roformer-chinese-char-small": {
            "vocab_size": 12000,
            "embedding_size": 384,
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 6,
            "intermediate_size": 1536,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "rotary_value": False,
        },
        "roformer-chinese-char-base": {
            "vocab_size": 12000,
            "embedding_size": 768,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "rotary_value": False,
        },
        "roformer-chinese-sim-char-ft-small": {
            "vocab_size": 12000,
            "embedding_size": 384,
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 6,
            "intermediate_size": 1536,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "rotary_value": False,
        },
        "roformer-chinese-sim-char-ft-base": {
            "vocab_size": 12000,
            "embedding_size": 768,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "rotary_value": False,
        },
        "roformer-chinese-sim-char-small": {
            "vocab_size": 12000,
            "embedding_size": 384,
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 6,
            "intermediate_size": 1536,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "rotary_value": False,
        },
        "roformer-chinese-sim-char-base": {
            "vocab_size": 12000,
            "embedding_size": 768,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "rotary_value": False,
        },
        "roformer-english-small-discriminator": {
            "vocab_size": 30522,
            "embedding_size": 128,
            "hidden_size": 256,
            "num_hidden_layers": 12,
            "num_attention_heads": 4,
            "intermediate_size": 1024,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 128,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "rotary_value": True,
        },
        "roformer-english-small-generator": {
            "vocab_size": 30522,
            "embedding_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 12,
            "num_attention_heads": 1,
            "intermediate_size": 256,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 128,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "rotary_value": True,
        },
    }

    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "roformer-chinese-small":
            "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-small/model_state.pdparams",
            "roformer-chinese-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-base/model_state.pdparams",
            "roformer-chinese-char-small":
            "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-char-small/model_state.pdparams",
            "roformer-chinese-char-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-char-base/model_state.pdparams",
            "roformer-chinese-sim-char-ft-small":
            "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-sim-char-ft-small/model_state.pdparams",
            "roformer-chinese-sim-char-ft-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-sim-char-ft-base/model_state.pdparams",
            "roformer-chinese-sim-char-small":
            "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-sim-char-small/model_state.pdparams",
            "roformer-chinese-sim-char-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-sim-char-base/model_state.pdparams",
            "roformer-english-small-discriminator":
            "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-english-small-discriminator/model_state.pdparams",
            "roformer-english-small-generator":
            "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-english-small-generator/model_state.pdparams",
        }
    }

    base_model_prefix = "roformer"

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.roformer.config["initializer_range"],
                        shape=layer.weight.shape, ))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class RoFormerModel(RoFormerPretrainedModel):
    """
    The bare RoFormer Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `RoFormerModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `RoFormerModel`.
        embedding_size (int, optional):
            Dimensionality of the embedding layer. Defaults to `768`.
        hidden_size (int, optional):
            Dimensionality of the, encoder layers and pooler layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `3072`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to `"gelu"`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids`.
            Defaults to `2`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer.
            Defaults to 0.02.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`BertPretrainedModel.init_weights()` for how weights are initialized in `BertModel`.

        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        pool_act (str, optional):
            The non-linear activation function in the pooler.
            Defaults to `"tanh"`.
        rotary_value (`bool`, optional):
            Whether or not apply rotay position embeddings to value.
            Defaults to `False`.
    """

    def __init__(
            self,
            vocab_size,
            embedding_size=768,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=1536,
            type_vocab_size=2,
            initializer_range=0.02,
            pad_token_id=0,
            pool_act="tanh",
            rotary_value=False, ):
        super(RoFormerModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        if embedding_size != hidden_size:
            self.embeddings_project = nn.Linear(embedding_size, hidden_size)
        self.embeddings = RoFormerEmbeddings(
            vocab_size,
            embedding_size,
            hidden_dropout_prob,
            type_vocab_size, )
        encoder_layer = TransformerEncoderLayerWithRotary(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
            rotary_value=rotary_value, )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = RoFormerPooler(hidden_size, pool_act)
        self.apply(self.init_weights)

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            output_hidden_states=False, ):
        r'''
        The RoFormerModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            position_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            output_hidden_states (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `False`.

        Returns:
            tuple: Returns tuple (`sequence_output`, `pooled_output`) or (`encoder_outputs`, `pooled_output`).

            With the fields:

            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].

            - `encoder_outputs` (List(Tensor)):
                A list of Tensor containing hidden-states of the model at each hidden layer in the Transformer encoder.
                The length of the list is `num_hidden_layers`.
                Each Tensor has a data type of float32 and its shape is [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerModel, RoFormerTokenizer

                tokenizer = RoFormerTokenizer.from_pretrained('roformer-chinese-base')
                model = RoFormerModel.from_pretrained('roformer-chinese-base')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        '''

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e4,
                axis=[1, 2], )
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids, )
        if hasattr(self, "embeddings_project"):
            embedding_output = self.embeddings_project(embedding_output)
        if output_hidden_states:
            output = embedding_output
            encoder_outputs = []
            for mod in self.encoder.layers:
                output = mod(output, src_mask=attention_mask)
                encoder_outputs.append(output)
            if self.encoder.norm is not None:
                encoder_outputs[-1] = self.encoder.norm(encoder_outputs[-1])
            pooled_output = self.pooler(encoder_outputs[-1])
        else:
            sequence_output = self.encoder(embedding_output, attention_mask)
            pooled_output = self.pooler(sequence_output)
        if output_hidden_states:
            return encoder_outputs, pooled_output
        else:
            return sequence_output, pooled_output


class RoFormerForQuestionAnswering(RoFormerPretrainedModel):
    """
    RoFormer with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        roformer (:class:`RoFormerModel`):
            An instance of RoFormerModel.
        dropout (float, optional):
            The dropout probability for output of RoFormer.
            If None, use the same value as `hidden_dropout_prob` of `RoFormerModel`
            instance `roformer`. Defaults to `None`.
        """

    def __init__(self, roformer, dropout=None):
        super(RoFormerForQuestionAnswering, self).__init__()
        self.roformer = roformer  # allow roformer to be config
        self.classifier = nn.Linear(self.roformer.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None):
        r"""
        The RoFormerForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`RoFormerModel`.
            token_type_ids (Tensor, optional):
                See :class:`RoFormerModel`.

        Returns:
            tuple: Returns tuple (`start_logits`, `end_logits`).

            With the fields:

            - `start_logits` (Tensor):
                A tensor of the input token classification logits, indicates the start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `end_logits` (Tensor):
                A tensor of the input token classification logits, indicates the end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerForQuestionAnswering
                from paddlenlp.transformers import RoFormerTokenizer

                tokenizer = RoFormerTokenizer.from_pretrained('roformer-chinese-base')
                model = RoFormerForQuestionAnswering.from_pretrained('roformer-chinese-base')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                start_logits = outputs[0]
                end_logits  =outputs[1]
        """
        sequence_output, _ = self.roformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=None, )

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class RoFormerForSequenceClassification(RoFormerPretrainedModel):
    """
    RoFormer Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        roformer (`RoFormerModel`):
            An instance of `paddlenlp.transformers.RoFormerModel`.
        num_classes (int, optional):
            The number of classes. Default to `2`.
        dropout (float, optional):
            The dropout probability for output of RoFormer.
            If None, use the same value as `hidden_dropout_prob`
            of `paddlenlp.transformers.RoFormerModel` instance. Defaults to `None`.
    """

    def __init__(self, roformer, num_classes=2, dropout=None):
        super(RoFormerForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.roformer = roformer  # allow roformer to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.roformer.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roformer.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`RoFormerModel`.
            token_type_ids (Tensor, optional):
                See :class:`RoFormerModel`.
            attention_mask (Tensor, optional):
                See :class:`RoFormerModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerForSequenceClassification, RoFormerTokenizer

                tokenizer = RoFormerTokenizer.from_pretrained('roformer-chinese-base')
                model = RoFormerForSequenceClassification.from_pretrained('roformer-chinese-base')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        _, pooled_output = self.roformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask, )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class RoFormerForTokenClassification(RoFormerPretrainedModel):
    """
    RoFormer Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        roformer (`RoFormerModel`):
            An instance of `paddlenlp.transformers.RoFormerModel`.
        num_classes (int, optional):
            The number of classes. Default to `2`.
        dropout (float, optional):
            The dropout probability for output of RoFormer.
            If None, use the same value as `hidden_dropout_prob`
            of `paddlenlp.transformers.RoFormerModel` instance. Defaults to `None`.
    """

    def __init__(self, roformer, num_classes=2, dropout=None):
        super(RoFormerForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.roformer = roformer  # allow roformer to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.roformer.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roformer.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`RoFormerModel`.
            token_type_ids (Tensor, optional):
                See :class:`RoFormerModel`.
            attention_mask (Tensor, optional):
                See :class:`RoFormerModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerForTokenClassification, RoFormerTokenizer

                tokenizer = RoFormerTokenizer.from_pretrained('roformer-chinese-base')
                model = RoFormerForTokenClassification.from_pretrained('roformer-chinese-base')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        sequence_output, _ = self.roformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask, )

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class RoFormerLMPredictionHead(Layer):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(RoFormerLMPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, embedding_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.decoder_weight = (self.create_parameter(
            shape=[vocab_size, embedding_size],
            dtype=self.transform.weight.dtype,
            is_bias=False, ) if embedding_weights is None else
                               embedding_weights)
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states,
                                           [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states,
                                                 masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = (paddle.tensor.matmul(
            hidden_states, self.decoder_weight, transpose_y=True) +
                         self.decoder_bias)
        return hidden_states


class RoFormerPretrainingHeads(Layer):
    """
    Perform language modeling task and next sentence classification task.

    Args:
        hidden_size (int):
            See :class:`RoFormerModel`.
        hidden_size (int):
            See :class:`RoFormerModel`.
        vocab_size (int):
            See :class:`RoFormerModel`.
        activation (str):
            Activation function used in the language modeling task.
        embedding_weights (Tensor, optional):
            Decoding weights used to map hidden_states to logits of the masked token prediction.
            Its data type should be float32 and its shape is [vocab_size, hidden_size].
            Defaults to `None`, which means use the same weights of the embedding layer.

    """

    def __init__(self,
                 embedding_size,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(RoFormerPretrainingHeads, self).__init__()
        self.predictions = RoFormerLMPredictionHead(embedding_size, hidden_size,
                                                    vocab_size, activation,
                                                    embedding_weights)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        """
        Args:
            sequence_output(Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].
            pooled_output(Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].
            masked_positions(Tensor, optional):
                A tensor indicates positions to be masked in the position embedding.
                Its data type should be int64 and its shape is [batch_size, mask_token_num].
                `mask_token_num` is the number of masked tokens. It should be no bigger than `sequence_length`.
                Defaults to `None`, which means we output hidden-states of all tokens in masked token prediction.

        Returns:
            tuple: Returns tuple (``prediction_scores``, ``seq_relationship_score``).

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size].

            - `seq_relationship_score` (Tensor):
                The scores of next sentence prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

        """
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class RoFormerForPretraining(RoFormerPretrainedModel):
    """
    RoFormer Model with pretraining tasks on top.

    Args:
        roformer (:class:`RoFormerModel`):
            An instance of :class:`RoFormerModel`.

    """

    def __init__(self, roformer):
        super(RoFormerForPretraining, self).__init__()
        self.roformer = roformer
        self.cls = RoFormerPretrainingHeads(
            self.roformer.config["embedding_size"],
            self.roformer.config["hidden_size"],
            self.roformer.config["vocab_size"],
            self.roformer.config["hidden_act"],
            embedding_weights=self.roformer.embeddings.word_embeddings.weight, )

        self.apply(self.init_weights)

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            masked_positions=None, ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`RoFormerModel`.
            token_type_ids (Tensor, optional):
                See :class:`RoFormerModel`.
            attention_mask (Tensor, optional):
                See :class:`RoFormerModel`.
            masked_positions(Tensor, optional):
                See :class:`RoFormerPretrainingHeads`.

        Returns:
            tuple: Returns tuple (``prediction_scores``, ``seq_relationship_score``).

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size].

            - `seq_relationship_score` (Tensor):
                The scores of next sentence prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

        """
        with paddle.static.amp.fp16_guard():
            outputs = self.roformer(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask, )
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_positions)
            return prediction_scores, seq_relationship_score


class RoFormerPretrainingCriterion(paddle.nn.Layer):
    """
    Args:
        vocab_size(int):
            Vocabulary size of `inputs_ids` in `RoFormerModel`. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling `RoFormerModel`.

    """

    def __init__(self, vocab_size):
        super(RoFormerPretrainingCriterion, self).__init__()
        # CrossEntropyLoss is expensive since the inner reshape (copy)
        self.loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(
            self,
            prediction_scores,
            seq_relationship_score,
            masked_lm_labels,
            next_sentence_labels,
            masked_lm_scale, ):
        """
        Args:
            prediction_scores(Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size]
            seq_relationship_score(Tensor):
                The scores of next sentence prediction. Its data type should be float32 and
                its shape is [batch_size, 2]
            masked_lm_labels(Tensor):
                The labels of the masked language modeling, its dimensionality is equal to `prediction_scores`.
                Its data type should be int64. If `masked_positions` is None, its shape is [batch_size, sequence_length, 1].
                Otherwise, its shape is [batch_size, mask_token_num, 1]
            next_sentence_labels(Tensor):
                The labels of the next sentence prediction task, the dimensionality of `next_sentence_labels`
                is equal to `seq_relation_labels`. Its data type should be int64 and
                its shape is [batch_size, 1]
            masked_lm_scale(Tensor or int):
                The scale of masked tokens. Used for the normalization of masked language modeling loss.
                If it is a `Tensor`, its data type should be int64 and its shape is equal to `prediction_scores`.

        Returns:
            Tensor: The pretraining loss, equals to the sum of `masked_lm_loss` plus the mean of `next_sentence_loss`.
            Its data type should be float32 and its shape is [1].

        """
        with paddle.static.amp.fp16_guard():
            masked_lm_loss = F.cross_entropy(
                prediction_scores,
                masked_lm_labels,
                reduction="none",
                ignore_index=-1)
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            next_sentence_loss = F.cross_entropy(
                seq_relationship_score, next_sentence_labels, reduction="none")
        return paddle.sum(masked_lm_loss) + paddle.mean(next_sentence_loss)
