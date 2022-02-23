# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .. import PretrainedModel, register_base_model

__all__ = [
    "MobileBertModel",
    "MobileBertPretrainedModel",
    "MobileBertForPreTraining",
    "MobileBertForSequenceClassification",
    "MobileBertForQuestionAnswering",
]

ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
}


class NoNorm(nn.Layer):
    def __init__(self, feat_size, eps=None):
        super().__init__()
        if isinstance(feat_size, int):
            feat_size = [feat_size]
        self.bias = paddle.create_parameter(feat_size, 'float32', is_bias=True)
        self.weight = paddle.create_parameter(
            feat_size,
            'float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0))

    def forward(self, input_tensor):
        return input_tensor * self.weight + self.bias


NORM2FN = {"layer_norm": nn.LayerNorm, "no_norm": NoNorm}


class MobileBertEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(
            self,
            vocab_size,
            embedding_size=128,
            hidden_size=512,
            hidden_dropout_prob=0.0,
            max_position_embeddings=512,
            type_vocab_size=2,
            layer_norm_eps=1e-12,
            pad_token_id=1,
            trigram_input=True,
            normalization_type="no_norm", ):
        super().__init__()
        self.trigram_input = trigram_input
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        embed_dim_multiplier = 3 if self.trigram_input else 1
        embedded_input_size = self.embedding_size * embed_dim_multiplier
        self.embedding_transformation = nn.Linear(embedded_input_size,
                                                  hidden_size)

        self.layer_norm = NORM2FN[normalization_type](hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids",
                             paddle.arange(max_position_embeddings).expand(
                                 (1, -1)))

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype='int64')
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.trigram_input:
            # From the paper MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited
            # Devices (https://arxiv.org/abs/2004.02984)
            #
            # The embedding table in BERT models accounts for a substantial proportion of model size. To compress
            # the embedding layer, we reduce the embedding dimension to 128 in MobileBERT.
            # Then, we apply a 1D convolution with kernel size 3 on the raw token embedding to produce a 512
            # dimensional output.
            inputs_embeds = paddle.concat(
                [
                    nn.functional.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0],
                                      value=0),
                    inputs_embeds,
                    nn.functional.pad(inputs_embeds[:, :-1],
                                      [0, 0, 1, 0, 0, 0],
                                      value=0),
                ],
                axis=2, )
        if self.trigram_input or self.embedding_size != self.hidden_size:
            inputs_embeds = self.embedding_transformation(inputs_embeds)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MobileBertAttention(nn.Layer):
    def __init__(
            self,
            num_attention_heads=4,
            true_hidden_size=128,
            hidden_size=512,
            use_bottleneck_attention=False,
            attention_probs_dropout_prob=0.1,
            use_bottleneck=True,
            normalization_type="no_norm",
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.0, ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(true_hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(true_hidden_size, self.all_head_size)
        self.key = nn.Linear(true_hidden_size, self.all_head_size)
        self.value = nn.Linear(true_hidden_size if use_bottleneck_attention else
                               hidden_size, self.all_head_size)

        self.attention_dropout = nn.Dropout(attention_probs_dropout_prob)

        self.use_bottleneck = use_bottleneck
        self.dense = nn.Linear(true_hidden_size, true_hidden_size)
        self.layer_norm = NORM2FN[normalization_type](true_hidden_size,
                                                      eps=layer_norm_eps)
        if not self.use_bottleneck:
            self.output_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads, self.attention_head_size
        ]
        x = x.reshape(new_x_shape)
        return x.transpose(perm=(0, 2, 1, 3))

    def forward(
            self,
            query_tensor,
            key_tensor,
            value_tensor,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=None, ):

        mixed_query_layer = self.query(query_tensor)
        mixed_key_layer = self.key(key_tensor)
        mixed_value_layer = self.value(value_tensor)

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
        attention_probs = nn.Softmax(axis=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(perm=(0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + [
            self.all_head_size
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)

        # dense layer shape to be checked
        projected_context_layer = self.dense(context_layer)

        # Run a linear projection of `hidden_size` then add a residual
        # with `hidden_states`.
        if not self.use_bottleneck:
            projected_context_layer = self.output_dropout(
                projected_context_layer)
        layer_normed_context_layer = self.layer_norm(hidden_states +
                                                     projected_context_layer)

        outputs = (layer_normed_context_layer,
                   attention_probs) if output_attentions else (
                       layer_normed_context_layer, )
        return outputs


class MobileBertIntermediate(nn.Layer):
    def __init__(
            self,
            true_hidden_size=128,
            intermediate_size=512,
            hidden_act="relu", ):
        super().__init__()
        self.dense = nn.Linear(true_hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class OutputBottleneck(nn.Layer):
    def __init__(
            self,
            true_hidden_size=128,
            hidden_size=512,
            normalization_type="no_norm",
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.0, ):
        super().__init__()
        self.dense = nn.Linear(true_hidden_size, hidden_size)
        self.layer_norm = NORM2FN[normalization_type](hidden_size,
                                                      eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, residual_tensor):
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.dropout(layer_outputs)
        layer_outputs = self.layer_norm(layer_outputs + residual_tensor)
        return layer_outputs


class MobileBertOutput(nn.Layer):
    def __init__(
            self,
            use_bottleneck=True,
            intermediate_size=512,
            true_hidden_size=128,
            hidden_size=512,
            normalization_type="no_norm",
            hidden_dropout_prob=0.0,
            layer_norm_eps=1e-12, ):
        super().__init__()
        self.use_bottleneck = use_bottleneck
        self.dense = nn.Linear(intermediate_size, true_hidden_size)
        self.layer_norm = NORM2FN[normalization_type](true_hidden_size)
        if not self.use_bottleneck:
            self.dropout = nn.Dropout(hidden_dropout_prob)
        else:
            self.bottleneck = OutputBottleneck(
                true_hidden_size=true_hidden_size,
                hidden_size=hidden_size,
                normalization_type=normalization_type,
                layer_norm_eps=layer_norm_eps,
                hidden_dropout_prob=hidden_dropout_prob, )

    def forward(self, intermediate_states, residual_tensor_1,
                residual_tensor_2):
        layer_output = self.dense(intermediate_states)
        if not self.use_bottleneck:
            layer_output = self.dropout(layer_output)
            layer_output = self.layer_norm(layer_output + residual_tensor_1)
        else:
            layer_output = self.layer_norm(layer_output + residual_tensor_1)
            layer_output = self.bottleneck(layer_output, residual_tensor_2)
        return layer_output


class BottleneckLayer(nn.Layer):
    def __init__(
            self,
            hidden_size=512,
            intra_bottleneck_size=128,
            normalization_type="no_norm",
            layer_norm_eps=1e-12, ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intra_bottleneck_size)
        self.layer_norm = NORM2FN[normalization_type](intra_bottleneck_size,
                                                      eps=layer_norm_eps)

    def forward(self, hidden_states):
        layer_input = self.dense(hidden_states)
        layer_input = self.layer_norm(layer_input)
        return layer_input


class Bottleneck(nn.Layer):
    def __init__(
            self,
            key_query_shared_bottleneck=True,
            use_bottleneck_attention=False,
            hidden_size=512,
            intra_bottleneck_size=128,
            normalization_type="no_norm",
            layer_norm_eps=1e-12, ):
        super().__init__()
        self.key_query_shared_bottleneck = key_query_shared_bottleneck
        self.use_bottleneck_attention = use_bottleneck_attention
        self.input = BottleneckLayer(
            hidden_size=hidden_size,
            intra_bottleneck_size=intra_bottleneck_size,
            normalization_type=normalization_type,
            layer_norm_eps=layer_norm_eps, )
        if self.key_query_shared_bottleneck:
            self.attention = BottleneckLayer(
                hidden_size=hidden_size,
                intra_bottleneck_size=intra_bottleneck_size,
                normalization_type=normalization_type,
                layer_norm_eps=layer_norm_eps, )

    def forward(self, hidden_states):
        # This method can return three different tuples of values. These different values make use of bottlenecks,
        # which are linear layers used to project the hidden states to a lower-dimensional vector, reducing memory
        # usage. These linear layer have weights that are learned during training.
        #
        # If `config.use_bottleneck_attention`, it will return the result of the bottleneck layer four times for the
        # key, query, value, and "layer input" to be used by the attention layer.
        # This bottleneck is used to project the hidden. This last layer input will be used as a residual tensor
        # in the attention self output, after the attention scores have been computed.
        #
        # If not `config.use_bottleneck_attention` and `config.key_query_shared_bottleneck`, this will return
        # four values, three of which have been passed through a bottleneck: the query and key, passed through the same
        # bottleneck, and the residual layer to be applied in the attention self output, through another bottleneck.
        #
        # Finally, in the last case, the values for the query, key and values are the hidden states without bottleneck,
        # and the residual layer will be this value passed through a bottleneck.

        bottlenecked_hidden_states = self.input(hidden_states)
        if self.use_bottleneck_attention:
            return (bottlenecked_hidden_states, ) * 4
        elif self.key_query_shared_bottleneck:
            shared_attention_input = self.attention(hidden_states)
            return (shared_attention_input, shared_attention_input,
                    hidden_states, bottlenecked_hidden_states)
        else:
            return (hidden_states, hidden_states, hidden_states,
                    bottlenecked_hidden_states)


class FFNOutput(nn.Layer):
    def __init__(
            self,
            intermediate_size=512,
            true_hidden_size=128,
            normalization_type="no_norm",
            layer_norm_eps=1e-12, ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, true_hidden_size)
        self.layer_norm = NORM2FN[normalization_type](true_hidden_size,
                                                      eps=layer_norm_eps)

    def forward(self, hidden_states, residual_tensor):
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.layer_norm(layer_outputs + residual_tensor)
        return layer_outputs


class FFNLayer(nn.Layer):
    def __init__(
            self,
            true_hidden_size=128,
            intermediate_size=512,
            hidden_act="relu",
            normalization_type="no_norm",
            layer_norm_eps=1e-12, ):
        super().__init__()
        self.intermediate = MobileBertIntermediate(
            true_hidden_size=true_hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act, )
        self.output = FFNOutput(
            intermediate_size=intermediate_size,
            true_hidden_size=true_hidden_size,
            normalization_type=normalization_type,
            layer_norm_eps=layer_norm_eps, )

    def forward(self, hidden_states):
        intermediate_output = self.intermediate(hidden_states)
        layer_outputs = self.output(intermediate_output, hidden_states)
        return layer_outputs


class MobileBertLayer(nn.Layer):
    def __init__(
            self,
            use_bottleneck=True,
            num_feedforward_networks=4,
            num_attention_heads=4,
            true_hidden_size=128,
            use_bottleneck_attention=False,
            attention_probs_dropout_prob=0.1,
            normalization_type="no_norm",
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.0,
            intermediate_size=512,
            hidden_act="relu",
            hidden_size=512,
            key_query_shared_bottleneck=True,
            intra_bottleneck_size=128, ):
        super().__init__()
        self.use_bottleneck = use_bottleneck
        self.num_feedforward_networks = num_feedforward_networks

        self.attention = MobileBertAttention(
            num_attention_heads=num_attention_heads,
            true_hidden_size=true_hidden_size,
            hidden_size=hidden_size,
            use_bottleneck_attention=use_bottleneck_attention,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_bottleneck=use_bottleneck,
            normalization_type=normalization_type,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob, )
        self.intermediate = MobileBertIntermediate(
            true_hidden_size=true_hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act, )
        self.output = MobileBertOutput(
            use_bottleneck=use_bottleneck,
            intermediate_size=intermediate_size,
            true_hidden_size=true_hidden_size,
            hidden_size=hidden_size,
            normalization_type=normalization_type,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps, )
        if self.use_bottleneck:
            self.bottleneck = Bottleneck(
                key_query_shared_bottleneck=key_query_shared_bottleneck,
                use_bottleneck_attention=use_bottleneck_attention,
                hidden_size=hidden_size,
                intra_bottleneck_size=intra_bottleneck_size,
                normalization_type=normalization_type,
                layer_norm_eps=layer_norm_eps, )
        if num_feedforward_networks > 1:
            self.ffn = nn.LayerList([
                FFNLayer(
                    true_hidden_size=true_hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=hidden_act,
                    normalization_type=normalization_type,
                    layer_norm_eps=layer_norm_eps, )
                for _ in range(num_feedforward_networks - 1)
            ])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=None, ):
        if self.use_bottleneck:
            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(
                hidden_states)
        else:
            query_tensor, key_tensor, value_tensor, layer_input = [
                hidden_states
            ] * 4

        self_attention_outputs = self.attention(
            query_tensor,
            key_tensor,
            value_tensor,
            layer_input,
            attention_mask,
            head_mask,
            output_attentions=output_attentions, )
        attention_output = self_attention_outputs[0]
        s = (attention_output, )
        outputs = self_attention_outputs[
            1:]  # add self attentions if we output attention weights

        if self.num_feedforward_networks != 1:
            for i, ffn_module in enumerate(self.ffn):
                attention_output = ffn_module(attention_output)
                s += (attention_output, )

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output,
                                   hidden_states)
        outputs = ((layer_output, ) + outputs + (
            paddle.to_tensor(1000),
            query_tensor,
            key_tensor,
            value_tensor,
            layer_input,
            attention_output,
            intermediate_output, ) + s)
        return outputs


class MobileBertEncoder(nn.Layer):
    def __init__(
            self,
            num_hidden_layers=24,
            use_bottleneck=True,
            num_feedforward_networks=4,
            num_attention_heads=4,
            true_hidden_size=128,
            use_bottleneck_attention=False,
            attention_probs_dropout_prob=0.1,
            normalization_type="no_norm",
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.0,
            intermediate_size=512,
            hidden_act="relu",
            hidden_size=512,
            key_query_shared_bottleneck=True, ):
        super().__init__()
        self.layers = nn.LayerList([
            MobileBertLayer(
                use_bottleneck=use_bottleneck,
                num_feedforward_networks=num_feedforward_networks,
                num_attention_heads=num_attention_heads,
                true_hidden_size=true_hidden_size,
                use_bottleneck_attention=use_bottleneck_attention,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                normalization_type=normalization_type,
                layer_norm_eps=layer_norm_eps,
                hidden_dropout_prob=hidden_dropout_prob,
                intermediate_size=intermediate_size,
                key_query_shared_bottleneck=key_query_shared_bottleneck,
                hidden_act=hidden_act,
                hidden_size=hidden_size, ) for _ in range(num_hidden_layers)
        ])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False, ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                output_attentions, )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        return tuple(
            v for v in [hidden_states, all_hidden_states, all_attentions]
            if v is not None)


class MobileBertPooler(nn.Layer):
    def __init__(
            self,
            classifier_activation=False,
            hidden_size=512, ):
        super().__init__()
        self.do_activate = classifier_activation
        if self.do_activate:
            self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        if not self.do_activate:
            return first_token_tensor
        else:
            pooled_output = self.dense(first_token_tensor)
            pooled_output = paddle.tanh(pooled_output)
            return pooled_output


class MobileBertPredictionHeadTransform(nn.Layer):
    def __init__(
            self,
            hidden_size=512,
            hidden_act="relu",
            layer_norm_eps=1e-12, ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, str):
            self.transform_act_fn = ACT2FN[hidden_act]
        else:
            self.transform_act_fn = hidden_act
        self.layer_norm = NORM2FN["layer_norm"](hidden_size,
                                                epsilon=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class MobileBertLMPredictionHead(nn.Layer):
    def __init__(
            self,
            vocab_size=30522,
            hidden_size=512,
            embedding_size=128,
            hidden_act="relu",
            layer_norm_eps=1e-12, ):
        super().__init__()
        self.transform = MobileBertPredictionHeadTransform(
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps, )
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.dense = nn.Linear(
            vocab_size, hidden_size - embedding_size, bias_attr=False)
        self.decoder = nn.Linear(embedding_size, vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        param_concat = paddle.concat(
            [self.decoder.weight, self.dense.weight.t()], axis=0)

        hidden_states = paddle.matmul(hidden_states, param_concat)
        hidden_states += self.decoder.bias
        return hidden_states


class MobileBertOnlyMLMHead(nn.Layer):
    def __init__(
            self,
            vocab_size=30522,
            hidden_size=512,
            embedding_size=128,
            hidden_act="relu",
            layer_norm_eps=1e-12, ):
        super().__init__()
        self.predictions = MobileBertLMPredictionHead(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps, )

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class MobileBertPreTrainingHeads(nn.Layer):
    def __init__(
            self,
            vocab_size=30522,
            hidden_size=512,
            embedding_size=128,
            hidden_act="relu",
            layer_norm_eps=1e-12, ):
        super().__init__()
        self.predictions = MobileBertLMPredictionHead(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps, )
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class MobileBertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained MobileBert models. It provides MobileBert related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "mobilebert-uncased": {
            "attention_probs_dropout_prob": 0.1,
            "classifier_activation": False,
            "embedding_size": 128,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 512,
            "intra_bottleneck_size": 128,
            "key_query_shared_bottleneck": True,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "normalization_type": "no_norm",
            "num_attention_heads": 4,
            "num_feedforward_networks": 4,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "trigram_input": True,
            "true_hidden_size": 128,
            "type_vocab_size": 2,
            "use_bottleneck": True,
            "use_bottleneck_attention": False,
            "vocab_size": 30522
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}

    pretrained_resource_files_map = {
        "model_state": {
            "mobilebert-uncased":
            "https://bj.bcebos.com/paddlenlp/models/transformers/mobilebert/mobilebert-uncased/model_state.pdparams"
        }
    }
    base_model_prefix = "mobilebert"

    def init_weights(self):
        # Initialize the weights.
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        # Initialize the weights.
        if isinstance(layer, nn.Linear):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.mobilebert.config["initializer_range"],
                    shape=layer.weight.shape))
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, (nn.LayerNorm, NoNorm)):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.ones_like(layer.weight))


class MobileBertForPreTraining(MobileBertPretrainedModel):
    """
    MobileBert Model with pretraining tasks on top.
    
    Args:
        bert (:class:`MobileBertModel`):
            An instance of :class:`MobileBertModel`.
    """

    def __init__(self, mobilebert):
        super(MobileBertForPreTraining, self).__init__()
        self.mobilebert = mobilebert
        self.cls = MobileBertPreTrainingHeads(
            self.mobilebert.config["vocab_size"],
            self.mobilebert.config["hidden_size"],
            self.mobilebert.config["embedding_size"],
            self.mobilebert.config["hidden_act"],
            self.mobilebert.config["layer_norm_eps"])

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddigs):
        self.cls.predictions.decoder = new_embeddigs

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None, ):
        r"""
        The MobileBertForPreTraining forward method, overrides the __call__() special method.
        
        Args:
            input_ids (Tensor):
                See :class:`MobileBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`MobileBertModel`.
            position_ids(Tensor, optional):
                See :class:`MobileBertModel`.
            head_mask (Tensor, optional):
                See :class:`MobileBertModel`.
            attention_mask (Tensor, optional):
                See :class:`MobileBertModel`.
            inputs_embeds (Tensor, optional):
                See :class:`MobileBertModel`.
            output_attentions (bool, optional):
                See :class:`MobileBertModel`.
            output_hidden_states (bool, optional):
                See :class:`MobileBertModel`.
        
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
        
        .. code-block::
                import paddle
                from paddlenlp.transformers import MobileBertModel, MobileBertTokenizer
                tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased')
                model = MobileBertForPreTraining.from_pretrained('mobilebert-uncased')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
                prediction_logits = outputs[0]
                seq_relationship_logits = outputs[1]
        """

        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output,
                                                             pooled_output)

        output = (prediction_scores, seq_relationship_score) + outputs[2:]
        return output


@register_base_model
class MobileBertModel(MobileBertPretrainedModel):
    """
    The bare MobileBert Model transformer outputting raw hidden-states.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.
    
    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `MobileBertModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `MobileBertModel`.
        embedding_size (int, optional):
            Embedding dimensionality of lookup_table in the embedding layer. Defaults to `128`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layer and pooler layer. Defaults to `512`.
        true_hidden_size (int, optional):
            Dimensionality of input_tensor in self attention layer. Defaults to `128`.
        use_bottleneck_attention (bool, optional):
            Using bottleneck to value tensor in self attention layer. Defaults to `False`.
        key_query_shared_bottleneck (bool, optional):
            Key and query shared bottleneck layer. Defaults to `True`. 
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `24`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `4`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `512`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to `"relu"`.
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
                See :meth:`MobileBertPretrainedModel.init_weights()` for how weights are initialized in `MobileBertModel`.
        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `1`.
        add_pooling_layer (bool, optional):
            Adding the pooling Layer after the encoder layer. Defaults to `True`.
        classifier_activation (bool, optional):
            Using the non-linear activation function in the pooling layer. Defaults to `False`.

    """

    def __init__(
            self,
            vocab_size,
            embedding_size=128,
            hidden_size=512,
            hidden_dropout_prob=0.0,
            max_position_embeddings=512,
            type_vocab_size=2,
            layer_norm_eps=1e-12,
            pad_token_id=1,
            trigram_input=True,
            normalization_type="no_norm",
            num_hidden_layers=24,
            use_bottleneck=True,
            num_feedforward_networks=4,
            num_attention_heads=4,
            true_hidden_size=128,
            use_bottleneck_attention=False,
            attention_probs_dropout_prob=0.1,
            intermediate_size=512,
            intra_bottleneck_size=128,
            hidden_act="relu",
            classifier_activation=False,
            initializer_range=0.02,
            key_query_shared_bottleneck=True,
            add_pooling_layer=True, ):
        super(MobileBertModel, self).__init__()

        self.initializer_range = initializer_range
        if use_bottleneck:
            true_hidden_size = intra_bottleneck_size
        else:
            true_hidden_size = hidden_size
        self.embeddings = MobileBertEmbeddings(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            trigram_input=trigram_input,
            normalization_type=normalization_type, )
        self.encoder = MobileBertEncoder(
            num_hidden_layers=num_hidden_layers,
            use_bottleneck=use_bottleneck,
            num_feedforward_networks=num_feedforward_networks,
            num_attention_heads=num_attention_heads,
            true_hidden_size=true_hidden_size,
            use_bottleneck_attention=use_bottleneck_attention,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            normalization_type=normalization_type,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_size=hidden_size,
            key_query_shared_bottleneck=key_query_shared_bottleneck, )
        self.num_hidden_layers = num_hidden_layers
        self.pooler = MobileBertPooler(
            classifier_activation=classifier_activation,
            hidden_size=hidden_size, ) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_head_mask(self,
                      head_mask,
                      num_hidden_layers,
                      is_attention_chunked=False):
        """
        Prepare the head mask if needed.
        
        Args:
            head_mask (:obj:`paddle.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.
        
        Returns:
            :obj:`paddle.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
            list with :obj:`[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask,
                                                      num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

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
        head_mask = head_mask.to(
            dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_hidden_states=None,
            output_attentions=None, ):
        r'''
        The MobileBertModel forward method, overrides the `__call__()` special method.
        
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
            head_mask (:obj:`paddle.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard). Defaults to `None`.
            output_hidden_states (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`.
            output_attentions (bool, optional):
                Whether to return the output of each self attention layers.
                Defaults to `None`.
        
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
                from paddlenlp.transformers import MobileBertModel, MobileBertTokenizer
                tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased')
                model = MobileBertModel.from_pretrained('mobilebert-uncased')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        '''

        output_attentions = output_attentions is not None
        output_hidden_states = (output_hidden_states is not None)

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
            attention_mask = paddle.ones(input_shape, dtype=input_ids.dtype)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype='int64')

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = attention_mask.unsqueeze(axis=[1, 2])
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(
            sequence_output) if self.pooler is not None else None

        return (sequence_output, pooled_output) + encoder_outputs[1:]


class MobileBertForSequenceClassification(MobileBertPretrainedModel):
    """
    MobileBert Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.
    
    Args:
        mobilebert (:class:`MobileBertModel`):
            An instance of MobileBert.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
    """

    def __init__(self, mobilebert, num_labels=2):
        super(MobileBertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.mobilebert = mobilebert
        classifier_dropout = (
            self.mobilebert.config["classifier_dropout"]
            if self.mobilebert.config.get("classifier_dropout") is not None else
            self.mobilebert.config["hidden_dropout_prob"])
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.mobilebert.config["hidden_size"],
                                    self.num_labels)

        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None):
        r"""
        The MobileBertForSequenceClassification forward method, overrides the __call__() special method.
        
        Args:
            input_ids (Tensor):
                See :class:`MobileBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`MobileBertModel`.
            position_ids(Tensor, optional):
                See :class:`MobileBertModel`.
            head_mask (Tensor, optional):
                See :class:`MobileBertModel`.
            attention_mask (Tensor, optional):
                See :class:`MobileBertModel`.
            inputs_embeds (Tensor, optional):
                See :class:`MobileBertModel`.
            output_attentions (bool, optional):
                See :class:`MobileBertModel`.
            output_hidden_states (bool, optional):
                See :class:`MobileBertModel`.
        
        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.
        
        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import MobileBertForSequenceClassification, MobileBertTokenizer
                tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased')
                model = MobileBertForSequenceClassification.from_pretrained('mobilebert-uncased', num_classes=2)
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
                print(logits.shape)
                # [1, 2]
        """

        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class MobileBertForQuestionAnswering(MobileBertPretrainedModel):
    """
    MobileBert Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.
    
    Args:
        mobilebert (:class:`MobileBert`):
            An instance of MobileBert.
    """

    def __init__(self, mobilebert):
        super(MobileBertForQuestionAnswering, self).__init__()
        self.num_labels = 2
        self.mobilebert = mobilebert
        self.qa_outputs = nn.Linear(self.mobilebert.config["hidden_size"],
                                    self.num_labels)

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
            output_attentions=None,
            output_hidden_states=None, ):
        r"""
        The MobileBertForQuestionAnswering forward method, overrides the __call__() special method.
        
        Args:
            input_ids (Tensor):
                See :class:`MobileBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`MobileBertModel`.
            position_ids(Tensor, optional):
                See :class:`MobileBertModel`.
            head_mask (Tensor, optional):
                See :class:`MobileBertModel`.
            attention_mask (Tensor, optional):
                See :class:`MobileBertModel`.
            inputs_embeds (Tensor, optional):
                See :class:`MobileBertModel`.
            output_attentions (bool, optional):
                See :class:`MobileBertModel`.
            output_hidden_states (bool, optional):
                See :class:`MobileBertModel`.
            start_positions (Tensor, optional):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
                sequence are not taken into account for computing the loss.
            end_positions (Tensor, optional):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
                sequence are not taken into account for computing the loss.
        
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
                from paddlenlp.transformers import MobileBertForQuestionAnswering, MobileBertTokenizer
                tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased')
                model = MobileBertForQuestionAnswering.from_pretrained('mobilebert-uncased')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)
                start_logits = outputs[0]
                end_logits = outputs[1]
        """
        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, )

        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)

        logits = paddle.transpose(logits, perm=[2, 0, 1])

        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        output = (start_logits, end_logits) + outputs[2:]

        return output
