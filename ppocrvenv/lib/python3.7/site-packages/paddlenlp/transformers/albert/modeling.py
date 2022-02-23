# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
"""Modeling classes for ALBERT model."""

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Layer
from .. import PretrainedModel, register_base_model

__all__ = [
    "AlbertPretrainedModel",
    "AlbertModel",
    "AlbertForPretraining",
    "AlbertForMaskedLM",
    "AlbertForSequenceClassification",
    "AlbertForTokenClassification",
    "AlbertForMultipleChoice",
]

dtype_float = paddle.get_default_dtype()


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(
            activation_string, list(ACT2FN.keys())))


def mish(x):
    return x * F.tanh(F.softplus(x))


def linear_act(x):
    return x


def swish(x):
    return x * F.sigmoid(x)


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + paddle.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))))


ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_new": gelu_new,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "mish": mish,
    "linear": linear_act,
    "swish": swish,
}


class AlbertEmbeddings(Layer):
    """
    Constructs the embeddings from word, position and token_type embeddings.
    """

    def __init__(
            self,
            vocab_size,
            embedding_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps,
            pad_token_id, ):
        super(AlbertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                embedding_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size,
                                                  embedding_size)

        self.layer_norm = nn.LayerNorm(embedding_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids",
                             paddle.arange(max_position_embeddings).expand(
                                 (1, -1)))

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0, ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:
                                             seq_length +
                                             past_key_values_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlbertAttention(Layer):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            layer_norm_eps, ):
        super(AlbertAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.attention_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)

    # Copied from transformers.models.bert.modeling_bert.BertSelfAttention.transpose_for_scores
    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads, self.attention_head_size
        ]
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False, ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = paddle.matmul(
            query_layer, key_layer, transpose_y=True)
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        context_layer = context_layer.reshape([0, 0, -1])

        # dense layer shape to be checked
        projected_context_layer = self.dense(context_layer)

        projected_context_layer_dropout = self.output_dropout(
            projected_context_layer)
        layer_normed_context_layer = self.layer_norm(
            hidden_states + projected_context_layer_dropout)
        return (layer_normed_context_layer,
                attention_probs) if output_attentions else (
                    layer_normed_context_layer, )


class AlbertLayer(Layer):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            layer_norm_eps, ):
        super(AlbertLayer, self).__init__()
        self.seq_len_dim = 1
        self.full_layer_layer_norm = nn.LayerNorm(
            hidden_size, epsilon=layer_norm_eps)
        self.attention = AlbertAttention(
            hidden_size,
            num_attention_heads,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            layer_norm_eps, )
        self.ffn = nn.Linear(hidden_size, intermediate_size)
        self.ffn_output = nn.Linear(intermediate_size, hidden_size)
        self.activation = ACT2FN[hidden_act]
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False, ):
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions, )

        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)

        hidden_states = self.full_layer_layer_norm(ffn_output +
                                                   attention_output[0])

        return (hidden_states,
                ) + attention_output[1:]  # add attentions if we output them


class AlbertLayerGroup(Layer):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            intermediate_size,
            inner_group_num,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            layer_norm_eps, ):
        super(AlbertLayerGroup, self).__init__()

        self.albert_layers = nn.LayerList([
            AlbertLayer(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                hidden_act,
                hidden_dropout_prob,
                attention_probs_dropout_prob,
                layer_norm_eps, ) for _ in range(inner_group_num)
        ])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            return_dict=False, ):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask,
                                        head_mask[layer_index], return_dict)
            hidden_states = layer_output[0]

            if return_dict:
                layer_attentions = layer_attentions + (layer_output[1], )
                layer_hidden_states = layer_hidden_states + (hidden_states, )

        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "all_hidden_states": layer_hidden_states,
                "all_attentions": layer_attentions,
            }
        return hidden_states


class AlbertTransformer(Layer):
    def __init__(
            self,
            embedding_size,
            hidden_size,
            num_hidden_layers,
            num_hidden_groups,
            num_attention_heads,
            intermediate_size,
            inner_group_num,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            layer_norm_eps, ):
        super(AlbertTransformer, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups

        self.embedding_hidden_mapping_in = nn.Linear(embedding_size,
                                                     hidden_size)
        self.albert_layer_groups = nn.LayerList([
            AlbertLayerGroup(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                inner_group_num,
                hidden_act,
                hidden_dropout_prob,
                attention_probs_dropout_prob,
                layer_norm_eps, ) for _ in range(num_hidden_groups)
        ])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            return_dict=False, ):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_hidden_states = (hidden_states, ) if return_dict else None
        all_attentions = () if return_dict else None

        for i in range(self.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.num_hidden_layers /
                                   self.num_hidden_groups)
            # Index of the hidden group
            group_idx = int(i /
                            (self.num_hidden_layers / self.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group:(group_idx + 1) *
                          layers_per_group],
                return_dict, )
            hidden_states = layer_group_output if not return_dict \
                else layer_group_output["last_hidden_state"]

            if return_dict:
                all_attentions = all_attentions + layer_group_output[
                    "all_attentions"]
                all_hidden_states = all_hidden_states + (hidden_states, )

        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "all_hidden_states": all_hidden_states,
                "all_attentions": all_attentions,
            }
        return hidden_states


class AlbertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained ALBERT models. It provides ALBERT related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "albert-base-v1": {
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 2,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "inner_group_num": 1,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_groups": 1,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30000
        },
        "albert-large-v1": {
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 2,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "inner_group_num": 1,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_groups": 1,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30000
        },
        "albert-xlarge-v1": {
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 2,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "inner_group_num": 1,
            "intermediate_size": 8192,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_groups": 1,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30000
        },
        "albert-xxlarge-v1": {
            "attention_probs_dropout_prob": 0,
            "bos_token_id": 2,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0,
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "inner_group_num": 1,
            "intermediate_size": 16384,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 64,
            "num_hidden_groups": 1,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30000
        },
        "albert-base-v2": {
            "attention_probs_dropout_prob": 0,
            "bos_token_id": 2,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "inner_group_num": 1,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_groups": 1,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30000
        },
        "albert-large-v2": {
            "attention_probs_dropout_prob": 0,
            "bos_token_id": 2,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "inner_group_num": 1,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_groups": 1,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30000
        },
        "albert-xlarge-v2": {
            "attention_probs_dropout_prob": 0,
            "bos_token_id": 2,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0,
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "inner_group_num": 1,
            "intermediate_size": 8192,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_groups": 1,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30000
        },
        "albert-xxlarge-v2": {
            "attention_probs_dropout_prob": 0,
            "bos_token_id": 2,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0,
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "inner_group_num": 1,
            "intermediate_size": 16384,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 64,
            "num_hidden_groups": 1,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30000
        },
        "albert-chinese-tiny": {
            "attention_probs_dropout_prob": 0.0,
            "bos_token_id": 2,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 312,
            "initializer_range": 0.02,
            "inner_group_num": 1,
            "intermediate_size": 1248,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_groups": 1,
            "num_hidden_layers": 4,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 21128
        },
        "albert-chinese-small": {
            "attention_probs_dropout_prob": 0.0,
            "bos_token_id": 2,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 384,
            "initializer_range": 0.02,
            "inner_group_num": 1,
            "intermediate_size": 1536,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_groups": 1,
            "num_hidden_layers": 6,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 21128
        },
        "albert-chinese-base": {
            "attention_probs_dropout_prob": 0,
            "bos_token_id": 2,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "inner_group_num": 1,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_groups": 1,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 21128
        },
        "albert-chinese-large": {
            "attention_probs_dropout_prob": 0,
            "bos_token_id": 2,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "inner_group_num": 1,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_groups": 1,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 21128
        },
        "albert-chinese-xlarge": {
            "attention_probs_dropout_prob": 0,
            "bos_token_id": 2,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0,
            "hidden_size": 2048,
            "initializer_range": 0.014,
            "inner_group_num": 1,
            "intermediate_size": 8192,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_groups": 1,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 21128
        },
        "albert-chinese-xxlarge": {
            "attention_probs_dropout_prob": 0,
            "bos_token_id": 2,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0,
            "hidden_size": 4096,
            "initializer_range": 0.01,
            "inner_group_num": 1,
            "intermediate_size": 16384,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_groups": 1,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 21128
        },
    }

    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "albert-base-v1":
            "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-base-v1.pdparams",
            "albert-large-v1":
            "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-large-v1.pdparams",
            "albert-xlarge-v1":
            "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-xlarge-v1.pdparams",
            "albert-xxlarge-v1":
            "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-xxlarge-v1.pdparams",
            "albert-base-v2":
            "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-base-v2.pdparams",
            "albert-large-v2":
            "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-large-v2.pdparams",
            "albert-xlarge-v2":
            "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-xlarge-v2.pdparams",
            "albert-xxlarge-v2":
            "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-xxlarge-v2.pdparams",
            "albert-chinese-tiny":
            "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-tiny.pdparams",
            "albert-chinese-small":
            "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-small.pdparams",
            "albert-chinese-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-base.pdparams",
            "albert-chinese-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-large.pdparams",
            "albert-chinese-xlarge":
            "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-xlarge.pdparams",
            "albert-chinese-xxlarge":
            "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-xxlarge.pdparams",
        }
    }
    base_model_prefix = "transformer"

    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        # Initialize the weights.
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.transformer.config["initializer_range"],
                    shape=layer.weight.shape))
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.transformer.config["initializer_range"],
                    shape=layer.weight.shape))
            if layer._padding_idx is not None:
                layer.weight[layer._padding_idx].set_value(
                    paddle.zeros_like(layer.weight[layer._padding_idx]))
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.ones_like(layer.weight))


@register_base_model
class AlbertModel(AlbertPretrainedModel):
    """
    The bare Albert Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in `AlbertModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `AlbertModel`.
            Defaults to `30000`.
        embedding_size (int, optional):
            Dimensionality of the embedding layer. Defaults to `128`.
        hidden_size (int, optional):
            Dimensionality of the encoder layer and pooler layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        inner_group_num (int, optional):
            Number of hidden groups in the Transformer encoder. Defaults to `1`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
        inner_group_num (int, optional):
            Number of inner groups in a hidden group. Default to `1`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids`. Defaults to `12`.

        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to `0.02`.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`BertPretrainedModel.init_weights()` for how weights are initialized in `ElectraModel`.

        layer_norm_eps(float, optional):
            The `epsilon` parameter used in :class:`paddle.nn.LayerNorm` for initializing layer normalization layers.
            A small value to the variance added to the normalization layer to prevent division by zero.
            Default to `1e-12`.
        pad_token_id (int, optional):
            The index of padding token in the token vocabulary. Defaults to `0`.
        add_pooling_layer(bool, optional):
            Whether or not to add the pooling layer. Default to `False`.
    """

    def __init__(
            self,
            vocab_size=30000,
            embedding_size=128,
            hidden_size=768,
            num_hidden_layers=12,
            num_hidden_groups=1,
            num_attention_heads=12,
            intermediate_size=3072,
            inner_group_num=1,
            hidden_act="gelu",
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
            add_pooling_layer=True, ):
        super(AlbertModel, self).__init__()
        self.initializer_range = initializer_range
        self.num_hidden_layers = num_hidden_layers
        self.embeddings = AlbertEmbeddings(
            vocab_size,
            embedding_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps,
            pad_token_id, )

        self.encoder = AlbertTransformer(
            embedding_size,
            hidden_size,
            num_hidden_layers,
            num_hidden_groups,
            num_attention_heads,
            intermediate_size,
            inner_group_num,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            layer_norm_eps, )

        if add_pooling_layer:
            self.pooler = nn.Linear(hidden_size, hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(
                -1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                -1)  # We can specify head_mask for each layer
        assert head_mask.dim(
        ) == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = paddle.cast(head_mask, dtype=dtype_float)
        return head_mask

    def get_head_mask(self,
                      head_mask,
                      num_hidden_layers,
                      is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask,
                                                      num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            return_dict=False, ):
        r'''
         The AlbertModel forward method, overrides the `__call__()` special method.

         Args:
             input_ids (Tensor):
                 Indices of input sequence tokens in the vocabulary. They are
                 numerical representations of tokens that build the input sequence.
                 Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
             attention_mask (Tensor, optional):
                 Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                 usually the paddings or the subsequent positions.
                 Its data type can be int, float and bool.
                 When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                 When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                 When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                 It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                 Defaults to `None`, which means nothing needed to be prevented attention to.
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
             head_mask (Tensor, optional):
                 Mask to nullify selected heads of the self-attention modules. Masks values can either be 0 or 1:

                 - 1 indicates the head is **not masked**,
                 - 0 indicated the head is **masked**.
             inputs_embeds (Tensor, optional):
                If you want to control how to convert `inputs_ids` indices into associated vectors, you can
                pass an embedded representation directly instead of passing `inputs_ids`.
             return_dict (bool, optional):
                 Whether or not to return a dict instead of a plain tuple. Default to `False`.


         Returns:
             tuple or Dict: Returns tuple (`sequence_output`, `pooled_output`) or a dict with
             `last_hidden_state`, `pooled_output`, `all_hidden_states`, `all_attentions` fields.

             With the fields:

             - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and has a shape of [`batch_size, sequence_length, hidden_size`].

             - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and
                has a shape of [batch_size, hidden_size].

             - `last_hidden_state` (Tensor):
                The output of the last encoder layer, it is also the `sequence_output`.
                It's data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].

             - `all_hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `all_hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

             - `all_attentions` (Tensor):
                Attentions of all layers of in the Transformer encoder. The length of `all_attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is
                [`batch_size, num_attention_heads, sequence_length, sequence_length`].

         Example:
             .. code-block::

                 import paddle
                 from paddlenlp.transformers import AlbertModel, AlbertTokenizer

                 tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
                 model = AlbertModel.from_pretrained('albert-base-v1')

                 inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                 inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                 output = model(**inputs)

         '''
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = paddle.ones(shape=input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(shape=input_shape, dtype="int64")

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = paddle.cast(
            extended_attention_mask, dtype=dtype_float)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds, )

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            return_dict=return_dict, )

        sequence_output = encoder_outputs if not return_dict \
            else encoder_outputs["last_hidden_state"]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) \
            if self.pooler is not None else None

        if return_dict:
            return {
                "last_hidden_state": sequence_output,
                "pooler_output": pooled_output,
                "all_hidden_states": encoder_outputs["all_hidden_states"],
                "all_attentions": encoder_outputs["all_attentions"],
            }
        return sequence_output, pooled_output


class AlbertForPretraining(AlbertPretrainedModel):
    """
    Albert Model with a `masked language modeling` head and a `sentence order prediction` head
    on top.

    Args:
        albert (:class:`AlbertModel`):
            An instance of :class:`AlbertModel`.
        lm_head (:class:`AlbertMLMHead`):
            An instance of :class:`AlbertSOPHead`.
        sop_head (:class:`AlbertSOPHead`):
            An instance of :class:`AlbertSOPHead`.
        vocab_size (int):
            See :class:`AlbertModel`.

    """

    def __init__(self, albert, lm_head, sop_head, vocab_size):
        super(AlbertForPretraining, self).__init__()

        self.transformer = albert
        self.predictions = lm_head
        self.sop_classifier = sop_head
        self.init_weights()
        self.vocab_size = vocab_size

    def get_output_embeddings(self):
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.predictions.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.embeddings.word_embeddings

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sentence_order_label=None,
            return_dict=False, ):
        r"""
        The AlbertForPretraining forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`AlbertModel`.
            attention_mask (list, optional):
                See :class:`AlbertModel`.
            token_type_ids (Tensor, optional):
                See :class:`AlbertModel`.
            position_ids(Tensor, optional):
                See :class:`AlbertModel`.
            head_mask(Tensor, optional):
                See :class:`AlbertModel`.
            inputs_embeds(Tensor, optional):
                See :class:`AlbertModel`.
            sentence_order_label(Tensor, optional):
                Labels of the next sequence prediction. Input should be a sequence pair
                Indices should be 0 or 1. ``0`` indicates original order (sequence A, then sequence B),
                and ``1`` indicates switched order (sequence B, then sequence A). Defaults to `None`.
            return_dict(bool, optional):
                See :class:`AlbertModel`.

        Returns:
            tuple or Dict: Returns tuple (`prediction_scores`, `sop_scores`) or a dict with
            `prediction_logits`, `sop_logits`, `pooled_output`, `hidden_states`, `attentions` fields.

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                and its shape is [batch_size, sequence_length, vocab_size].

            - `sop_scores` (Tensor):
                The scores of sentence order prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

            - `prediction_logits` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                and its shape is [batch_size, sequence_length, vocab_size].

            - `sop_logits` (Tensor):
                The scores of sentence order prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

            - `hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

            - `attentions` (Tensor):
                Attentions of all layers of in the Transformer encoder. The length of `attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is
                [`batch_size, num_attention_heads, sequence_length, sequence_length`].

        """

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict, )
        sequence_output = outputs[0] if not return_dict \
            else outputs["last_hidden_state"]
        pooled_output = outputs[1] if not return_dict \
            else outputs["pooler_output"]

        prediction_scores = self.predictions(sequence_output)
        sop_scores = self.sop_classifier(pooled_output)

        if return_dict:
            return {
                "prediction_logits": prediction_scores,
                "sop_logits": sop_scores,
                "hidden_states": outputs["all_hidden_states"],
                "attentions": outputs["all_attentions"],
            }
        return prediction_scores, sop_scores


class AlbertMLMHead(Layer):
    def __init__(
            self,
            embedding_size,
            vocab_size,
            hidden_size,
            hidden_act, ):
        super(AlbertMLMHead, self).__init__()

        self.layer_norm = nn.LayerNorm(embedding_size)
        self.bias = self.create_parameter(
            [vocab_size],
            is_bias=True,
            default_initializer=nn.initializer.Constant(value=0))
        self.dense = nn.Linear(hidden_size, embedding_size)
        self.decoder = nn.Linear(embedding_size, vocab_size)
        self.activation = ACT2FN[hidden_act]

        # link bias
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        prediction_scores = hidden_states
        return prediction_scores


class AlbertSOPHead(Layer):
    def __init__(
            self,
            classifier_dropout_prob,
            hidden_size,
            num_labels, ):
        super(AlbertSOPHead, self).__init__()
        self.dropout = nn.Dropout(classifier_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output):
        dropout_pooled_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_pooled_output)
        return logits


class AlbertForMaskedLM(AlbertPretrainedModel):
    """
    Albert Model with a `masked language modeling` head on top.

    Args:
        albert (:class:`AlbertModel`):
            An instance of :class:`AlbertModel`.

    """

    def __init__(self, albert):
        super(AlbertForMaskedLM, self).__init__()

        self.transformer = albert
        self.predictions = AlbertMLMHead(
            embedding_size=self.transformer.config["embedding_size"],
            vocab_size=self.transformer.config["vocab_size"],
            hidden_size=self.transformer.config["hidden_size"],
            hidden_act=self.transformer.config["hidden_act"], )
        self.init_weights()

    def get_output_embeddings(self):
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.predictions.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.embeddings.word_embeddings

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            return_dict=False, ):
        r"""
        The AlbertForPretraining forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`AlbertModel`.
            attention_mask (list, optional):
                See :class:`AlbertModel`.
            token_type_ids (Tensor, optional):
                See :class:`AlbertModel`.
            position_ids(Tensor, optional):
                See :class:`AlbertModel`.
            head_mask(Tensor, optional):
                See :class:`AlbertModel`.
            inputs_embeds(Tensor, optional):
                See :class:`AlbertModel`.
            return_dict(bool, optional):
                See :class:`AlbertModel`.

        Returns:
            Tensor or Dict: Returns tensor `prediction_scores` or a dict with `logits`,
            `hidden_states`, `attentions` fields.

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                and its shape is [batch_size, sequence_length, vocab_size].

            - `logits` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                and its shape is [batch_size, sequence_length, vocab_size].

            - `hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

            - `attentions` (Tensor):
                Attentions of all layers of in the Transformer encoder. The length of `attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is
                [`batch_size, num_attention_heads, sequence_length, sequence_length`].

        """

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict, )

        sequence_outputs = transformer_outputs[0] if not return_dict \
            else transformer_outputs["last_hidden_state"]
        prediction_scores = self.predictions(sequence_outputs)

        if return_dict:
            return {
                "logits": prediction_scores,
                "hidden_states": transformer_outputs["all_hidden_states"],
                "attentions": transformer_outputs["all_attentions"]
            }
        return prediction_scores


class AlbertForSequenceClassification(AlbertPretrainedModel):
    """
    Albert Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        albert (:class:`AlbertModel`):
            An instance of AlbertModel.
        classifier_dropput_prob (float, optional):
            The dropout probability for the classifier.
            Defaults to `0`.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.

    """

    def __init__(self, albert, classifier_dropout_prob=0, num_classes=2):
        super(AlbertForSequenceClassification, self).__init__()
        self.num_classes = num_classes

        self.transformer = albert
        self.dropout = nn.Dropout(classifier_dropout_prob)
        self.classifier = nn.Linear(self.transformer.config["hidden_size"],
                                    self.num_classes)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            return_dict=False, ):
        r"""
        The AlbertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`AlbertModel`.
            attention_mask (list, optional):
                See :class:`AlbertModel`.
            token_type_ids (Tensor, optional):
                See :class:`AlbertModel`.
            position_ids(Tensor, optional):
                See :class:`AlbertModel`.
            head_mask(Tensor, optional):
                See :class:`AlbertModel`.
            inputs_embeds(Tensor, optional):
                See :class:`AlbertModel`.
            return_dict(bool, optional):
                See :class:`AlbertModel`.

        Returns:
            Tensor or Dict: Returns tensor `logits`, or a dict with `logits`, `hidden_states`, `attentions` fields.

            With the fields:

            - `logits` (Tensor):
                A tensor of the input text classification logits.
                Shape as `[batch_size, num_classes]` and dtype as float32.

            - `hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

            - `attentions` (Tensor):
                Attentions of all layers of in the Transformer encoder. The length of `attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is
                [`batch_size, num_attention_heads, sequence_length, sequence_length`].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import AlbertForSequenceClassification, AlbertTokenizer

                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
                model = AlbertForSequenceClassification.from_pretrained('albert-base-v1')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict, )

        pooled_output = transformer_outputs[1] if not return_dict \
            else transformer_outputs["pooler_output"]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if return_dict:
            return {
                "logits": logits,
                "hidden_states": transformer_outputs["all_hidden_states"],
                "attentions": transformer_outputs["all_attentions"]
            }
        return logits


class AlbertForTokenClassification(AlbertPretrainedModel):
    """
    Albert Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        albert (:class:`AlbertModel`):
            An instance of AlbertModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.

    """

    def __init__(self, albert, num_classes=2):
        super(AlbertForTokenClassification, self).__init__()
        self.num_classes = num_classes

        self.transformer = albert
        self.dropout = nn.Dropout(self.transformer.config[
            "hidden_dropout_prob"])
        self.classifier = nn.Linear(self.transformer.config["hidden_size"],
                                    self.num_classes)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            return_dict=False, ):
        r"""
        The AlbertForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`AlbertModel`.
            attention_mask (list, optional):
                See :class:`AlbertModel`.
            token_type_ids (Tensor, optional):
                See :class:`AlbertModel`.
            position_ids(Tensor, optional):
                See :class:`AlbertModel`.
            head_mask(Tensor, optional):
                See :class:`AlbertModel`.
            inputs_embeds(Tensor, optional):
                See :class:`AlbertModel`.
            return_dict(bool, optional):
                See :class:`AlbertModel`.

        Returns:
            Tensor or Dict: Returns tensor `logits`, or a dict with `logits`, `hidden_states`, `attentions` fields.

            With the fields:

            - `logits` (Tensor):
                A tensor of the input token classification logits.
                Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

            - `hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

            - `attentions` (Tensor):
                Attentions of all layers of in the Transformer encoder. The length of `attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is
                [`batch_size, num_attention_heads, sequence_length, sequence_length`].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import AlbertForTokenClassification, AlbertTokenizer

                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
                model = AlbertForTokenClassification.from_pretrained('albert-base-v1')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict, )

        sequence_output = transformer_outputs[0] if not return_dict \
            else transformer_outputs["sequence_output"]
        logits = self.classifier(sequence_output)

        if return_dict:
            return {
                "logits": logits,
                "hidden_states": transformer_outputs["all_hidden_states"],
                "attentions": transformer_outputs["all_attentions"]
            }
        return logits


class AlbertForQuestionAnswering(AlbertPretrainedModel):
    """
    Albert Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        albert (:class:`AlbertModel`):
            An instance of AlbertModel.
        num_classes (int):
            The number of classes.

    """

    def __init__(self, albert, num_labels):
        super(AlbertForQuestionAnswering, self).__init__()
        self.num_labels = num_labels
        self.transformer = albert

        self.qa_outputs = nn.Linear(self.transformer.config["hidden_size"],
                                    num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            return_dict=False, ):
        r"""
        The AlbertForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`AlbertModel`.
            attention_mask (list, optional):
                See :class:`AlbertModel`.
            token_type_ids (Tensor, optional):
                See :class:`AlbertModel`.
            position_ids(Tensor, optional):
                See :class:`AlbertModel`.
            head_mask(Tensor, optional):
                See :class:`AlbertModel`.
            inputs_embeds(Tensor, optional):
                See :class:`AlbertModel`.
            start_positions(Tensor, optional):
                Start positions of the text. Defaults to `None`.
            end_positions(Tensor, optional):
                End positions of the text. Defaults to `None`.
            return_dict(bool, optional):
                See :class:`AlbertModel`.

        Returns:
            tuple or Dict: Returns tuple (`start_logits, end_logits`)or a dict
            with `start_logits`, `end_logits`, `hidden_states`, `attentions` fields.

            With the fields:

            - `start_logits` (Tensor):
                A tensor of the input token classification logits, indicates the start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `end_logits` (Tensor):
                A tensor of the input token classification logits, indicates the end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

            - `attentions` (Tensor):
                Attentions of all layers of in the Transformer encoder. The length of `attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is
                [`batch_size, num_attention_heads, sequence_length, sequence_length`].


        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import AlbertForQuestionAnswering, AlbertTokenizer

                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
                model = AlbertForQuestionAnswering.from_pretrained('albert-base-v1')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict, )
        sequence_output = transformer_outputs[0] if not return_dict \
            else transformer_outputs["sequence_output"]
        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = paddle.split(
            logits, num_or_sections=1, axis=-1)
        start_logits = start_logits.squeeze(axis=-1)
        end_logits = start_logits.squeeze(axis=-1)

        if return_dict:
            return {
                "start_logits": start_logits,
                "end_logits": end_logits,
                "hidden_states": transformer_outputs["all_hidden_states"],
                "attentions": transformer_outputs["all_attentions"]
            }
        return start_logits, end_logits


class AlbertForMultipleChoice(AlbertPretrainedModel):
    """
    Albert Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like SWAG tasks .

    Args:
        albert (:class:`AlbertModel`):
            An instance of AlbertModel.

    """

    def __init__(self, albert):
        super(AlbertForMultipleChoice, self).__init__()
        self.transformer = albert
        self.dropout = nn.Dropout(self.transformer.config[
            "hidden_dropout_prob"])
        self.classifier = nn.Linear(self.transformer.config["hidden_size"], 1)
        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            return_dict=False, ):
        r"""
        The AlbertForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`AlbertModel`.
            attention_mask (list, optional):
                See :class:`AlbertModel`.
            token_type_ids (Tensor, optional):
                See :class:`AlbertModel`.
            position_ids(Tensor, optional):
                See :class:`AlbertModel`.
            head_mask(Tensor, optional):
                See :class:`AlbertModel`.
            inputs_embeds(Tensor, optional):
                See :class:`AlbertModel`.
            start_positions(Tensor, optional):
                Start positions of the text. Defaults to `None`.
            end_positions(Tensor, optional):
                End positions of the text. Defaults to `None`.
            return_dict(bool, optional):
                See :class:`AlbertModel`.

        Returns:
            Tensor or Dict: Returns tensor `reshaped_logits` or a dict
            with `reshaped_logits`, `hidden_states`, `attentions` fields.

            With the fields:

            - `reshaped_logits` (Tensor):
                A tensor of the input multiple choice classification logits.
                Shape as `[batch_size, num_classes]` and dtype as `float32`.

            - `hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

            - `attentions` (Tensor):
                Attentions of all layers of in the Transformer encoder. The length of `attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is
                [`batch_size, num_attention_heads, sequence_length, sequence_length`].
        """

        num_choices = input_ids.shape[
            1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.reshape([-1, input_ids.shape[-1]]) \
            if input_ids is not None else None
        attention_mask = attention_mask.reshape([-1, attention_mask.shape[-1]]) \
            if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape([-1, token_type_ids.shape[-1]]) \
            if token_type_ids is not None else None
        position_ids = position_ids.reshape([-1, position_ids.shape[-1]]) \
            if position_ids is not None else None
        inputs_embeds = (inputs_embeds.reshape(
            [-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1]])
                         if inputs_embeds is not None else None)
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict, )
        pooled_output = transformer_outputs[1] if not return_dict \
            else transformer_outputs["pooler_output"]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape([-1, num_choices])

        if return_dict:
            return {
                "logits": reshaped_logits,
                "hidden_states": transformer_outputs["all_hidden_states"],
                "attentions": transformer_outputs["all_attentions"]
            }
        return reshaped_logits
