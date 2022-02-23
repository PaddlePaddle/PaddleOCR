# coding=utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The SqueezeBert authors and The HuggingFace Inc. team.
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
import paddle
from paddle import nn
from .. import PretrainedModel, register_base_model

__all__ = [
    'SqueezeBertModel',
    'SqueezeBertForSequenceClassification',
    'SqueezeBertForTokenClassification',
    'SqueezeBertForQuestionAnswering',
]

ACT2FN = {'gelu': nn.GELU()}


def _convert_attention_mask(attention_mask, inputs):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask.unsqueeze(1)
    elif attention_mask.dim() == 2:
        # extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
    extended_attention_mask = paddle.cast(extended_attention_mask,
                                          inputs.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class SqueezeBertEmbeddings(nn.Layer):
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 layer_norm_eps=1e-12):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=None)

        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.register_buffer("position_ids",
                             paddle.arange(max_position_embeddings).expand(
                                 (1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        input_shape = input_ids.shape
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(
                input_shape,
                dtype=paddle.int64, )

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MatMulWrapper(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, mat1, mat2):
        """
        :param inputs: two paddle tensors :return: matmul of these tensors
        Here are the typical dimensions found in BERT (the B is optional) mat1.shape: [B, <optional extra dims>, M, K]
        mat2.shape: [B, <optional extra dims>, K, N] output shape: [B, <optional extra dims>, M, N]
        """
        return paddle.matmul(mat1, mat2)


class SqueezeBertLayerNorm(nn.LayerNorm):
    def __init__(self, hidden_size, epsilon=1e-12):
        nn.LayerNorm.__init__(
            self, normalized_shape=hidden_size,
            epsilon=epsilon)  # instantiates self.{weight, bias, eps}

    def forward(self, x):
        x = x.transpose((0, 2, 1))
        x = nn.LayerNorm.forward(self, x)
        return x.transpose((0, 2, 1))


class ConvDropoutLayerNorm(nn.Layer):
    def __init__(self, cin, cout, groups, dropout_prob):
        super().__init__()

        self.conv1d = nn.Conv1D(
            in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
        self.layernorm = SqueezeBertLayerNorm(cout)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        x = self.conv1d(hidden_states)
        x = self.dropout(x)
        x = x + input_tensor
        x = self.layernorm(x)
        return x


class ConvActivation(nn.Layer):
    def __init__(self, cin, cout, groups, act):
        super().__init__()
        self.conv1d = nn.Conv1D(
            in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
        self.act = ACT2FN[act]

    def forward(self, x):
        output = self.conv1d(x)
        return self.act(output)


class SqueezeBertSelfAttention(nn.Layer):
    def __init__(self,
                 num_attention_heads,
                 attention_probs_dropout_prob,
                 cin,
                 q_groups=1,
                 k_groups=1,
                 v_groups=1):
        super().__init__()
        if cin % num_attention_heads != 0:
            raise ValueError(
                f"cin ({cin}) is not a multiple of the number of attention heads ({num_attention_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(cin / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Conv1D(
            in_channels=cin, out_channels=cin, kernel_size=1, groups=q_groups)
        self.key = nn.Conv1D(
            in_channels=cin, out_channels=cin, kernel_size=1, groups=k_groups)
        self.value = nn.Conv1D(
            in_channels=cin, out_channels=cin, kernel_size=1, groups=v_groups)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.softmax = nn.Softmax(axis=-1)

        self.matmul_qk = MatMulWrapper()
        self.matmul_qkv = MatMulWrapper()

    def transpose_for_scores(self, x):
        """
        - input: [N, C, W]
        - output: [N, C1, W, C2] where C1 is the head index, and C2 is one head's contents
        """
        new_x_shape = (x.shape[0], self.num_attention_heads,
                       self.attention_head_size, x.shape[-1])  # [N, C1, C2, W]
        x = x.reshape(new_x_shape)
        return x.transpose((0, 1, 3, 2))  # [N, C1, C2, W] --> [N, C1, W, C2]

    def transpose_key_for_scores(self, x):
        """
        - input: [N, C, W]
        - output: [N, C1, C2, W] where C1 is the head index, and C2 is one head's contents
        """
        new_x_shape = (x.shape[0], self.num_attention_heads,
                       self.attention_head_size, x.shape[-1])  # [N, C1, C2, W]
        x = x.reshape(new_x_shape)
        return x

    def transpose_output(self, x):
        """
        - input: [N, C1, W, C2]
        - output: [N, C, W]
        """
        x = x.transpose((0, 1, 3, 2))  # [N, C1, C2, W]
        new_x_shape = (x.shape[0], self.all_head_size, x.shape[3])  # [N, C, W]
        x = x.reshape(new_x_shape)
        return x

    def forward(self, hidden_states, attention_mask, output_attentions):
        """
        expects hidden_states in [N, C, W] data layout.
        The attention_mask data layout is [N, W], and it does not need to be transposed.
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_score = self.matmul_qk(query_layer, key_layer)
        attention_score = attention_score / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_score = attention_score + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_score)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = self.matmul_qkv(attention_probs, value_layer)
        context_layer = self.transpose_output(context_layer)

        result = {"context_layer": context_layer}
        if output_attentions:
            result["attention_score"] = attention_score
        return result


class SqueezeBertLayer(nn.Layer):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads,
                 attention_probs_dropout_prob, q_groups, k_groups, v_groups,
                 output_groups, intermediate_groups, post_attention_groups,
                 hidden_dropout_prob, hidden_act):
        """
        - hidden_size = input chans = output chans for Q, K, V (they are all the same ... for now) = output chans for
          the module
        - intermediate_size = output chans for intermediate layer
        - groups = number of groups for all layers in the BertLayer. (eventually we could change the interface to
          allow different groups for different layers)
        """
        super().__init__()

        c0 = hidden_size
        c1 = hidden_size
        c2 = intermediate_size
        c3 = hidden_size

        self.attention = SqueezeBertSelfAttention(
            num_attention_heads,
            attention_probs_dropout_prob,
            cin=c0,
            q_groups=q_groups,
            k_groups=k_groups,
            v_groups=v_groups)
        self.post_attention = ConvDropoutLayerNorm(
            cin=c0,
            cout=c1,
            groups=post_attention_groups,
            dropout_prob=hidden_dropout_prob)
        self.intermediate = ConvActivation(
            cin=c1, cout=c2, groups=intermediate_groups, act=hidden_act)
        self.output = ConvDropoutLayerNorm(
            cin=c2,
            cout=c3,
            groups=output_groups,
            dropout_prob=hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask, output_attentions):
        att = self.attention(hidden_states, attention_mask, output_attentions)
        attention_output = att["context_layer"]

        post_attention_output = self.post_attention(attention_output,
                                                    hidden_states)
        intermediate_output = self.intermediate(post_attention_output)
        layer_output = self.output(intermediate_output, post_attention_output)

        output_dict = {"feature_map": layer_output}
        if output_attentions:
            output_dict["attention_score"] = att["attention_score"]

        return output_dict


class SqueezeBertEncoder(nn.Layer):
    def __init__(self, embedding_size, hidden_size, intermediate_size,
                 num_attention_heads, attention_probs_dropout_prob, q_groups,
                 k_groups, v_groups, output_groups, intermediate_groups,
                 post_attention_groups, hidden_dropout_prob, hidden_act,
                 num_hidden_layers):
        super().__init__()
        assert embedding_size == hidden_size, (
            "If you want embedding_size != intermediate hidden_size,"
            "please insert a Conv1D layer to adjust the number of channels "
            "before the first SqueezeBertLayer.")
        self.layers = nn.LayerList(
            SqueezeBertLayer(hidden_size, intermediate_size,
                             num_attention_heads, attention_probs_dropout_prob,
                             q_groups, k_groups, v_groups, output_groups,
                             intermediate_groups, post_attention_groups,
                             hidden_dropout_prob, hidden_act)
            for _ in range(num_hidden_layers))

    def forward(self,
                hidden_states,
                attention_mask=None,
                output_attentions=False,
                output_hidden_states=False):

        hidden_states = hidden_states.transpose((0, 2, 1))
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                hidden_states = hidden_states.transpose((0, 2, 1))
                all_hidden_states += (hidden_states, )
                hidden_states = hidden_states.transpose((0, 2, 1))

            layer_output = layer.forward(hidden_states, attention_mask,
                                         output_attentions)

            hidden_states = layer_output["feature_map"]

            if output_attentions:
                all_attentions += (layer_output["attention_score"], )

        # [batch_size, hidden_size, sequence_length] --> [batch_size, sequence_length, hidden_size]
        hidden_states = hidden_states.transpose((0, 2, 1))

        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        return tuple(
            v for v in [hidden_states, all_hidden_states, all_attentions]
            if v is not None)


class SqueezeBertPooler(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SqueezeBertPredictionHeadTransform(nn.Layer):
    def __init__(self, hidden_size, hidden_act, layer_norm_eps):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, str):
            self.transform_act_fn = ACT2FN[hidden_act]
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class SqueezeBertLMPredictionHead(nn.Layer):
    def __init__(self, hidden_size, hidden_act, layer_norm_eps, vocab_size):
        super().__init__()
        self.transform = SqueezeBertPredictionHeadTransform(
            hidden_size, hidden_act, layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias_attr=False)
        self.bias = paddle.create_parameter(
            [vocab_size], dtype='float32', is_bias=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class SqueezeBertPreTrainingHeads(nn.Layer):
    def __init__(self, hidden_size, hidden_act, layer_norm_eps, vocab_size):
        super().__init__()
        self.predictions = SqueezeBertLMPredictionHead(
            hidden_size, hidden_act, layer_norm_eps, vocab_size)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class SqueezeBertPreTrainedModel(PretrainedModel):
    """
        An abstract class for pretrained SqueezBert models. It provides SqueezBert related
        `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
        `pretrained_init_configuration`, `base_model_prefix` for downloading and
        loading pretrained models. See `PretrainedModel` for more details.
    """

    base_model_prefix = "squeezebert"
    model_config_file = "model_config.json"

    pretrained_init_configuration = {
        "squeezebert-uncased": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "model_type": "squeezebert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 30528,
            "q_groups": 4,
            "k_groups": 4,
            "v_groups": 4,
            "post_attention_groups": 1,
            "intermediate_groups": 4,
            "output_groups": 4,
            "pad_token_id": 0,
            'layer_norm_eps': 1e-12
        },
        "squeezebert-mnli": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "model_type": "squeezebert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 30528,
            "q_groups": 4,
            "k_groups": 4,
            "v_groups": 4,
            "post_attention_groups": 1,
            "intermediate_groups": 4,
            "output_groups": 4,
            "num_labels": 3,
            "pad_token_id": 0,
            'layer_norm_eps': 1e-12
        },
        "squeezebert-mnli-headless": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "model_type": "squeezebert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 30528,
            "q_groups": 4,
            "k_groups": 4,
            "v_groups": 4,
            "post_attention_groups": 1,
            "intermediate_groups": 4,
            "output_groups": 4,
            "pad_token_id": 0,
            'layer_norm_eps': 1e-12
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}

    pretrained_resource_files_map = {
        "model_state": {
            "squeezebert-uncased":
            "http://bj.bcebos.com/paddlenlp/models/transformers/squeezebert/squeezebert-uncased/model_state.pdparams",
            "squeezebert-mnli":
            "http://bj.bcebos.com/paddlenlp/models/transformers/squeezebert/squeezebert-mnli/model_state.pdparams",
            "squeezebert-mnli-headless":
            "http://bj.bcebos.com/paddlenlp/models/transformers/squeezebert/squeezebert-mnli-headless/model_state.pdparams",
        }
    }

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.squeezebert.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class SqueezeBertModel(SqueezeBertPreTrainedModel):
    """
    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `SqueezeBertModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `BertModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layer and pooler layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `12`.
        intermediate_size (int, optional):
            Output chans for intermediate layer.
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
            Defaults to `16`.
        q_groups (int)
            number of query groups for all layers in the BertModule. (eventually we could change the interface to
            allow different groups for different layers)
        k_groups (int)
            number of key groups for all layers in the BertModule. (eventually we could change the interface to
            allow different groups for different layers)
        v_groups (int)
            number of value groups for all layers in the BertModule. (eventually we could change the interface to
            allow different groups for different layers)
        output_groups (int)
            number of output groups for all layers in the BertModule. (eventually we could change the interface to
            allow different groups for different layers)
        intermediate_groups (int)
            number of intermediate groups for all layers in the BertModule. (eventually we could change the interface to
            allow different groups for different layers)
        post_groups (int)
            number of post groups for all layers in the BertModule. (eventually we could change the interface to
            allow different groups for different layers)
        initializer_range (float, optional):
            The standard deviation of the normal initializer.
            Defaults to 0.02.
            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`BertPretrainedModel.init_weights()` for how weights are initialized in `BertModel`.
    """

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads,
                 intermediate_size,
                 hidden_act,
                 hidden_dropout_prob,
                 attention_probs_dropout_prob,
                 max_position_embeddings,
                 type_vocab_size,
                 q_groups,
                 k_groups,
                 v_groups,
                 output_groups,
                 intermediate_groups,
                 post_attention_groups,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super().__init__()
        self.initializer_range = initializer_range
        self.embeddings = SqueezeBertEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size, layer_norm_eps)
        self.encoder = SqueezeBertEncoder(
            embedding_size, hidden_size, intermediate_size, num_attention_heads,
            attention_probs_dropout_prob, q_groups, k_groups, v_groups,
            output_groups, intermediate_groups, post_attention_groups,
            hidden_dropout_prob, hidden_act, num_hidden_layers)
        self.pooler = SqueezeBertPooler(hidden_size)

        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                output_attentions=None,
                output_hidden_states=None):
        r'''
        The  forward method, overrides the `__call__()` special method.
        Args:
           input_ids (Tensor):
               Indices of input sequence tokens in the vocabulary. They are
               numerical representations of tokens that build the input sequence.
               Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
           attention_mask (Tensor, optional):
               Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
               usually the paddings or the subsequent positions.
               Its data type can be int, float and bool.
               If its data type is int, the values should be either 0 or 1.
               - **1** for tokens that **not masked**,
               - **0** for tokens that **masked**.
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

           output_attentions (bool, optional):
               Whether to return the attention_weight of each hidden layers.
               Defaults to `False`.
           output_hidden_states (bool, optional):
               Whether to return the output of each hidden layers.
               Defaults to `False`.
        Returns:
           tuple: Returns tuple (`sequence_output`, `pooled_output`) with (`encoder_outputs`, `encoder_attentions`) by
           optional.
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
               The length of the list is `num_hidden_layers` + 1 (Embedding Layer output).
               Each Tensor has a data type of float32 and its shape is [batch_size, sequence_length, hidden_size].
        '''
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        extended_attention_mask = _convert_attention_mask(attention_mask,
                                                          embedding_output)
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return (sequence_output, pooled_output) + encoder_outputs[1:]


class SqueezeBertForSequenceClassification(SqueezeBertPreTrainedModel):
    """
    SqueezeBert Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
    for GLUE tasks.
    Args:
        squeezebert (:class:`SqueezeBertModel`):
            An instance of SqueezeBert.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of SqueezeBertModel.
            If None, use the same value as `hidden_dropout_prob` of `SqueezeBertModel`
            instance `squeezebert`. Defaults to None.
    """

    def __init__(self, squeezebert, num_classes=2, dropout=None):
        super().__init__()
        self.num_classes = num_classes
        self.squeezebert = squeezebert
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  squeezebert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.squeezebert.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        The SqueezeBertForSequenceClassification forward method, overrides the __call__() special method.
        Args:
            input_ids (Tensor):
                See :class:`SqueezeBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`SqueezeBertModel`.
            position_ids(Tensor, optional):
                See :class:`SqueezeBertModel`.
            attention_mask (list, optional):
                See :class:`SqueezeBertModel`.
        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.
        """

        _, pooled_output = self.squeezebert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class SqueezeBertForQuestionAnswering(SqueezeBertPreTrainedModel):
    """
    SqueezeBert Model with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and
    `span end logits`).
    Args:
        squeezebert (:class:`SqueezeBertModel`):
            An instance of SqueezeBertModel.
        dropout (float, optional):
            The dropout probability for output of SqueezeBert.
            If None, use the same value as `hidden_dropout_prob` of `SqueezeBertModel`
            instance `squeezebert`. Defaults to `None`.
        """

    def __init__(self, squeezebert, dropout=None):
        super().__init__()
        self.squeezebert = squeezebert
        self.classifier = nn.Linear(self.squeezebert.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None):
        r'''
        The SqueezeBertForQuestionAnswering forward method, overrides the __call__() special method.
        Args:
            input_ids (Tensor):
                See :class:`SqueezeBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`SqueezeBertModel`.
        Returns:
            tuple: Returns tuple (`start_logits`, `end_logits`).
            With the fields:
            - `start_logits` (Tensor):
                A tensor of the input token classification logits, indicates the start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].
            - `end_logits` (Tensor):
                A tensor of the input token classification logits, indicates the end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].
        '''
        sequence_output, _ = self.squeezebert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=None,
            attention_mask=None)
        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)
        return start_logits, end_logits


class SqueezeBertForTokenClassification(SqueezeBertPreTrainedModel):
    """
    SqueezeBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    Args:
        squeezebert (:class:`SqueezeBertModel`):
            An instance of SqueezeBertModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of squeezebert.
            If None, use the same value as `hidden_dropout_prob` of `SqueezeBert`
            instance `squeezebert`. Defaults to None.
    """

    def __init__(self, squeezebert, num_classes=2, dropout=None):
        super().__init__()
        self.num_classes = num_classes
        self.squeezebert = squeezebert
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  squeezebert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.squeezebert.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        The SqueezeBertForTokenClassification forward method, overrides the __call__() special method.
        Args:
            input_ids (Tensor):
                See :class:`SqueezeBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`SqueezeBertModel`.
            position_ids(Tensor, optional):
                See :class:`SqueezeBertModel`.
            attention_mask (list, optional):
                See :class:`SqueezeBertModel`.
        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.
        """

        sequence_output, _ = self.squeezebert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits
