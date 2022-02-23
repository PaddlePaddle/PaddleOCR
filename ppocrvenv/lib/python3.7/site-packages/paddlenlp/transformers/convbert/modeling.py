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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import tensor
from paddle.nn import Layer
from ..electra.modeling import get_activation
from .. import PretrainedModel, register_base_model

__all__ = [
    "ConvBertModel", "ConvBertPretrainedModel", "ConvBertForTotalPretraining",
    "ConvBertDiscriminator", "ConvBertGenerator", "ConvBertClassificationHead",
    "ConvBertForSequenceClassification", "ConvBertForTokenClassification",
    "ConvBertPretrainingCriterion", "ConvBertForQuestionAnswering",
    "ConvBertForMultipleChoice", "ConvBertForPretraining"
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


class GroupedLinear(nn.Layer):
    def __init__(self, input_size, output_size, num_groups):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_groups = num_groups
        self.group_in_dim = self.input_size // self.num_groups
        self.group_out_dim = self.output_size // self.num_groups
        self.weight = paddle.create_parameter(
            shape=[self.num_groups, self.group_in_dim, self.group_out_dim],
            dtype=dtype_float)
        self.bias = paddle.create_parameter(
            shape=[output_size], dtype=dtype_float, is_bias=True)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        x = tensor.reshape(hidden_states,
                           [-1, self.num_groups, self.group_in_dim])
        x = tensor.transpose(x, perm=[1, 0, 2])
        x = tensor.matmul(x, self.weight)
        x = tensor.transpose(x, perm=[1, 0, 2])
        x = tensor.reshape(x, [batch_size, -1, self.output_size])
        x = x + self.bias
        return x


class SeparableConv1D(nn.Layer):
    """This class implements separable convolution, i.e. a depthwise and a pointwise layer"""

    def __init__(self, input_filters, output_filters, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv1D(
            input_filters,
            input_filters,
            kernel_size=kernel_size,
            groups=input_filters,
            padding=kernel_size // 2,
            bias_attr=False,
            data_format="NLC", )
        self.pointwise = nn.Conv1D(
            input_filters,
            output_filters,
            kernel_size=1,
            bias_attr=False,
            data_format="NLC", )
        self.bias = paddle.create_parameter(
            shape=[output_filters], dtype=dtype_float, is_bias=True)

    def forward(self, hidden_states):
        x = self.depthwise(hidden_states)
        x = self.pointwise(x) + self.bias
        return x


class MultiHeadAttentionWithConv(Layer):
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.,
            kdim=None,
            vdim=None,
            need_weights=False,
            conv_kernel_size=None,
            head_ratio=None, ):
        super(MultiHeadAttentionWithConv, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.need_weights = need_weights
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        new_num_attention_heads = num_heads // head_ratio
        if num_heads // head_ratio < 1:
            self.num_heads = 1
            self.conv_type = "noconv"
        else:
            self.num_heads = new_num_attention_heads
            self.conv_type = "sdconv"

        self.all_head_size = self.num_heads * self.head_dim

        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(embed_dim, self.all_head_size)
        self.k_proj = nn.Linear(self.kdim, self.all_head_size)
        self.v_proj = nn.Linear(self.vdim, self.all_head_size)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if self.conv_type == "sdconv":
            self.conv_kernel_size = conv_kernel_size
            self.key_conv_attn_layer = SeparableConv1D(
                embed_dim, self.all_head_size, self.conv_kernel_size)
            self.conv_kernel_layer = nn.Linear(
                self.all_head_size, self.num_heads * self.conv_kernel_size)
            self.conv_out_layer = nn.Linear(embed_dim, self.all_head_size)
            self.unfold = nn.Unfold(
                kernel_sizes=[self.conv_kernel_size, 1],
                paddings=[(self.conv_kernel_size - 1) // 2, 0], )

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if self.conv_type == "sdconv":
            mixed_key_conv_attn_layer = self.key_conv_attn_layer(query)
            conv_attn_layer = mixed_key_conv_attn_layer * q
            batch_size = q.shape[0]
            # conv_kernel_layer
            conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
            conv_kernel_layer = tensor.reshape(
                conv_kernel_layer, shape=[-1, self.conv_kernel_size, 1])
            conv_kernel_layer = F.softmax(conv_kernel_layer, axis=1)
            # conv_out
            conv_out_layer = self.conv_out_layer(query)
            conv_out_layer = tensor.reshape(
                conv_out_layer, [batch_size, -1, self.all_head_size, 1])
            conv_out_layer = tensor.transpose(conv_out_layer, perm=[0, 2, 1, 3])
            conv_out_layer = self.unfold(conv_out_layer)
            conv_out_layer = tensor.transpose(conv_out_layer, perm=[0, 2, 1])
            conv_out_layer = tensor.reshape(
                conv_out_layer,
                shape=[-1, self.head_dim, self.conv_kernel_size])
            conv_out_layer = tensor.matmul(conv_out_layer, conv_kernel_layer)
            conv_out = tensor.reshape(
                conv_out_layer,
                shape=[batch_size, -1, self.num_heads, self.head_dim])

        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])

        product = tensor.matmul(x=q, y=k, transpose_y=True) * self.scale
        if attn_mask is not None:
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask

        weights = F.softmax(product)
        weights = self.dropout(weights)
        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        if self.conv_type == "sdconv":
            out = tensor.concat([out, conv_out], axis=2)
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerEncoderLayerWithConv(nn.TransformerEncoderLayer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 conv_kernel_size=None,
                 head_ratio=None,
                 num_groups=None,
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
        self.self_attn = MultiHeadAttentionWithConv(
            d_model,
            nhead,
            dropout=attn_dropout,
            conv_kernel_size=conv_kernel_size,
            head_ratio=head_ratio, )
        if num_groups > 1:
            self.linear1 = GroupedLinear(
                d_model, dim_feedforward, num_groups=num_groups)
            self.linear2 = GroupedLinear(
                dim_feedforward, d_model, num_groups=num_groups)
        self._config.update({
            "conv_kernel_size": conv_kernel_size,
            "head_ratio": head_ratio,
            "num_groups": num_groups
        })


class ConvBertEmbeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(
            self,
            vocab_size,
            embedding_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size, ):
        super(ConvBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                embedding_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size,
                                                  embedding_size)

        self.layer_norm = nn.LayerNorm(embedding_size, epsilon=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True

        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ConvBertDiscriminatorPredictions(nn.Layer):
    """
    Prediction layer for the discriminator.
    """

    def __init__(self, hidden_size, hidden_act):
        super(ConvBertDiscriminatorPredictions, self).__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dense_prediction = nn.Linear(hidden_size, 1)
        self.act = get_activation(hidden_act)

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.act(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()

        return logits


class ConvBertGeneratorPredictions(nn.Layer):
    """
    Prediction layer for the generator.
    """

    def __init__(self, embedding_size, hidden_size, hidden_act):
        super(ConvBertGeneratorPredictions, self).__init__()

        self.layer_norm = nn.LayerNorm(embedding_size)
        self.dense = nn.Linear(hidden_size, embedding_size)
        self.act = get_activation(hidden_act)

    def forward(self, generator_hidden_states):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class ConvBertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained ConvBert models. It provides ConvBert related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    base_model_prefix = "convbert"
    model_config_file = "model_config.json"

    # pretrained general configuration
    gen_weight = 1.0
    disc_weight = 50.0
    tie_word_embeddings = True
    untied_generator_embeddings = False
    use_softmax_sample = True

    # model init configuration
    pretrained_init_configuration = {
        "convbert-base": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522,
            "conv_kernel_size": 9,
            "head_ratio": 2,
            "num_groups": 1,
        },
        "convbert-medium-small": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 128,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 384,
            "initializer_range": 0.02,
            "intermediate_size": 1536,
            "max_position_embeddings": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522,
            "conv_kernel_size": 9,
            "head_ratio": 2,
            "num_groups": 2,
        },
        "convbert-small": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 128,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "max_position_embeddings": 512,
            "num_attention_heads": 4,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522,
            "conv_kernel_size": 9,
            "head_ratio": 2,
            "num_groups": 1,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "convbert-base":
            "http://bj.bcebos.com/paddlenlp/models/transformers/convbert/convbert-base/model_state.pdparams",
            "convbert-medium-small":
            "http://bj.bcebos.com/paddlenlp/models/transformers/convbert/convbert-medium-small/model_state.pdparams",
            "convbert-small":
            "http://bj.bcebos.com/paddlenlp/models/transformers/convbert/convbert-small/model_state.pdparams",
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
        if hasattr(self, "get_output_embeddings") and hasattr(
                self, "get_input_embeddings"):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings,
                                           self.get_input_embeddings())

    def _init_weights(self, layer):
        """ Initialize the weights """
        if isinstance(layer, (nn.Linear, nn.Embedding, GroupedLinear)):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.convbert.config["initializer_range"],
                    shape=layer.weight.shape, ))
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.full_like(layer.weight, 1.0))
            layer._epsilon = 1e-12
        elif isinstance(layer, SeparableConv1D):
            layer.depthwise.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.convbert.config["initializer_range"],
                    shape=layer.depthwise.weight.shape, ))
            layer.pointwise.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.convbert.config["initializer_range"],
                    shape=layer.pointwise.weight.shape, ))

        if isinstance(layer, (nn.Linear, GroupedLinear,
                              SeparableConv1D)) and layer.bias is not None:
            layer.bias.set_value(paddle.zeros_like(layer.bias))

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """ Tie or clone layer weights """
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


@register_base_model
class ConvBertModel(ConvBertPretrainedModel):
    """
    The bare ConvBert Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `ConvBertModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `ConvBertModel`.
        embedding_size (int, optional):
            Dimensionality of the embedding layer. Defaults to `768`.
        hidden_size (int, optional):
            Dimensionality of the encoder layer and pooler layer. Defaults to `768`.
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
                See :meth:`ConvBertPretrainedModel.init_weights()` for how weights are initialized in `ConvBertModel`.

        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.

        conv_kernel_size (int, optional):
            The size of the convolutional kernel.
            Defaults to `9`.

        head_ratio (int, optional):
            Ratio gamma to reduce the number of attention heads.
            Defaults to `2`.

        num_groups (int, optional):
            The number of groups for grouped linear layers for ConvBert model.
            Defaults to `1`.

    """

    def __init__(self,
                 vocab_size,
                 embedding_size=768,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 pad_token_id=0,
                 conv_kernel_size=9,
                 head_ratio=2,
                 num_groups=1):
        super(ConvBertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = ConvBertEmbeddings(
            vocab_size,
            embedding_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size, )

        if embedding_size != hidden_size:
            self.embeddings_project = nn.Linear(embedding_size, hidden_size)

        encoder_layer = TransformerEncoderLayerWithConv(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
            conv_kernel_size=conv_kernel_size,
            head_ratio=head_ratio,
            num_groups=num_groups, )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r'''
        The ConvBertModel forward method, overrides the `__call__()` special method.

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
                If its data type is int, the values should be either 0 or 1.

                - **1** for tokens that **not masked**,
                - **0** for tokens that **masked**.

                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.

        Returns:
            Tensor: Returns Tensor `sequence_output`, sequence of hidden-states at the last layer of the model.
            Shape as `[batch_size, sequence_length, hidden_size]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ConvBertModel, ConvBertTokenizer

                tokenizer = ConvBertTokenizer.from_pretrained('convbert-base')
                model = ConvBertModel.from_pretrained('convbert-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        '''
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(dtype_float) * -1e4,
                axis=[1, 2])
        else:
            attention_mask = paddle.unsqueeze(
                attention_mask, axis=[1, 2]).astype(dtype_float)
            attention_mask = (1.0 - attention_mask) * -1e4

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids, )

        if hasattr(self, "embeddings_project"):
            embedding_output = self.embeddings_project(embedding_output)

        sequence_output = self.encoder(embedding_output, attention_mask)

        return sequence_output


class ConvBertDiscriminator(ConvBertPretrainedModel):
    """
    ConvBert Model with a discriminator prediction head on top.

    Args:
        convbert (:class:`ConvBertModel`):
            An instance of ConvBertModel.
    """

    def __init__(self, convbert):
        super(ConvBertDiscriminator, self).__init__()

        self.convbert = convbert
        self.discriminator_predictions = ConvBertDiscriminatorPredictions(
            self.convbert.config["hidden_size"],
            self.convbert.config["hidden_act"])
        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r'''
        The ConvBertDiscriminator forward method, overrides the `__call__()` special method.

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
                If its data type is int, the values should be either 0 or 1.

                - **1** for tokens that **not masked**,
                - **0** for tokens that **masked**.

                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the discriminator prediction logits.
            Shape as `[batch_size, sequence_length]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ConvBertDiscriminatorPredictions, ConvBertTokenizer

                tokenizer = ConvBertTokenizer.from_pretrained('convbert-base')
                model = ConvBertDiscriminatorPredictions.from_pretrained('convbert-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
        '''

        discriminator_sequence_output = self.convbert(
            input_ids, token_type_ids, position_ids, attention_mask)

        logits = self.discriminator_predictions(discriminator_sequence_output)

        return logits


class ConvBertGenerator(ConvBertPretrainedModel):
    """
    ConvBert Model with a generator prediction head on top.

    Args:
        convbert (:class:`ConvBertModel`):
            An instance of ConvBertModel.
    """

    def __init__(self, convbert):
        super(ConvBertGenerator, self).__init__()

        self.convbert = convbert
        self.generator_predictions = ConvBertGeneratorPredictions(
            self.convbert.config["embedding_size"],
            self.convbert.config["hidden_size"],
            self.convbert.config["hidden_act"], )

        if not self.tie_word_embeddings:
            self.generator_lm_head = nn.Linear(
                self.convbert.config["embedding_size"],
                self.convbert.config["vocab_size"])
        else:
            self.generator_lm_head_bias = paddle.create_parameter(
                shape=[self.convbert.config["vocab_size"]],
                dtype=dtype_float,
                is_bias=True, )
        self.init_weights()

    def get_input_embeddings(self):
        return self.convbert.embeddings.word_embeddings

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            attention_mask=None, ):
        r'''
        The ConvBertGenerator forward method, overrides the `__call__()` special method.

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
                If its data type is int, the values should be either 0 or 1.

                - **1** for tokens that **not masked**,
                - **0** for tokens that **masked**.

                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.

        Returns:
            Tensor: Returns tensor `prediction_scores`, a tensor of the generator prediction scores.
            Shape as `[batch_size, sequence_length, vocab_size]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ConvBertGenerator, ConvBertTokenizer

                tokenizer = ConvBertTokenizer.from_pretrained('convbert-base')
                model = ConvBertGenerator.from_pretrained('convbert-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                prediction_scores = model(**inputs)
        '''

        generator_sequence_output = self.convbert(input_ids, token_type_ids,
                                                  position_ids, attention_mask)

        prediction_scores = self.generator_predictions(
            generator_sequence_output)
        if not self.tie_word_embeddings:
            prediction_scores = self.generator_lm_head(prediction_scores)
        else:
            prediction_scores = paddle.add(
                paddle.matmul(
                    prediction_scores,
                    self.get_input_embeddings().weight,
                    transpose_y=True, ),
                self.generator_lm_head_bias, )

        return prediction_scores


class ConvBertClassificationHead(nn.Layer):
    """
    ConvBert head for sentence-level classification tasks.
    """

    def __init__(self, hidden_size, hidden_dropout_prob, num_classes):
        super(ConvBertClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_classes)
        self.act = get_activation("gelu")

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take [CLS] token
        x = self.dropout(x)
        x = self.dense(x)
        x = self.act(x)  # ConvBert paper used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ConvBertForSequenceClassification(ConvBertPretrainedModel):
    """
    ConvBert Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        convbert (:class:`ConvBertModel`):
            An instance of ConvBertModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of ConvBert.
            If None, use the same value as `hidden_dropout_prob` of `ConvBertModel`
            instance `convbert`. Defaults to None.
    """

    def __init__(self, convbert, num_classes=2, dropout=None):
        super(ConvBertForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.convbert = convbert
        self.classifier = ConvBertClassificationHead(
            hidden_size=self.convbert.config["hidden_size"],
            hidden_dropout_prob=dropout if dropout is not None else
            self.convbert.config["hidden_dropout_prob"],
            num_classes=self.num_classes, )

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            attention_mask=None, ):
        r"""
        The ConvBertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ConvBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`ConvBertModel`.
            position_ids(Tensor, optional):
                See :class:`ConvBertModel`.
            attention_mask (list, optional):
                See :class:`ConvBertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ConvBertForSequenceClassification, ConvBertTokenizer

                tokenizer = ConvBertTokenizer.from_pretrained('convbert-base')
                model = ConvBertForSequenceClassification.from_pretrained('convbert-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
        """

        sequence_output = self.convbert(input_ids, token_type_ids, position_ids,
                                        attention_mask)

        logits = self.classifier(sequence_output)

        return logits


class ConvBertForTokenClassification(ConvBertPretrainedModel):
    """
    ConvBert Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        convbert (:class:`ConvBertModel`):
            An instance of ConvBertModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of ConvBert.
            If None, use the same value as `hidden_dropout_prob` of `ConvBertModel`
            instance `convbert`. Defaults to None.
    """

    def __init__(self, convbert, num_classes=2, dropout=None):
        super(ConvBertForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.convbert = convbert
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.convbert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.convbert.config["hidden_size"],
                                    self.num_classes)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            attention_mask=None, ):
        r"""
        The ConvBertForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ConvBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`ConvBertModel`.
            position_ids(Tensor, optional):
                See :class:`ConvBertModel`.
            attention_mask (list, optional):
                See :class:`ConvBertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ConvBertForTokenClassification, ConvBertTokenizer

                tokenizer = ConvBertTokenizer.from_pretrained('convbert-base')
                model = ConvBertForTokenClassification.from_pretrained('convbert-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
        """

        sequence_output = self.convbert(input_ids, token_type_ids, position_ids,
                                        attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits


class ConvBertForTotalPretraining(ConvBertPretrainedModel):
    """
    Combine generator with discriminator for Replaced Token Detection (RTD) pretraining.
    """
    pretrained_init_configuration = {
        "convbert-base-generator": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "max_position_embeddings": 512,
            "num_attention_heads": 4,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522,
            "conv_kernel_size": 9,
            "head_ratio": 2,
            "num_groups": 1,
        },
        "convbert-medium-small-generator": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 128,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 96,
            "initializer_range": 0.02,
            "intermediate_size": 384,
            "max_position_embeddings": 512,
            "num_attention_heads": 2,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522,
            "conv_kernel_size": 9,
            "head_ratio": 2,
            "num_groups": 2,
        },
        "convbert-small-generator": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 128,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 64,
            "initializer_range": 0.02,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "num_attention_heads": 1,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522,
            "conv_kernel_size": 9,
            "head_ratio": 2,
            "num_groups": 1,
        },
        "convbert-base-discriminator": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522,
            "conv_kernel_size": 9,
            "head_ratio": 2,
            "num_groups": 1,
        },
        "convbert-medium-small-discriminator": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 128,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 384,
            "initializer_range": 0.02,
            "intermediate_size": 1536,
            "max_position_embeddings": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522,
            "conv_kernel_size": 9,
            "head_ratio": 2,
            "num_groups": 2,
        },
        "convbert-small-discriminator": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 128,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "max_position_embeddings": 512,
            "num_attention_heads": 4,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522,
            "conv_kernel_size": 9,
            "head_ratio": 2,
            "num_groups": 1,
        },
    }

    def __init__(self, generator, discriminator):
        super(ConvBertForTotalPretraining, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.initializer_range = discriminator.convbert.initializer_range
        self.init_weights()

    def get_input_embeddings(self):
        if not self.untied_generator_embeddings:
            return self.generator.convbert.embeddings.word_embeddings
        else:
            return None

    def get_output_embeddings(self):
        if not self.untied_generator_embeddings:
            return self.discriminator.convbert.embeddings.word_embeddings
        else:
            return None

    def get_discriminator_inputs(self, inputs, raw_inputs, gen_logits,
                                 gen_labels, use_softmax_sample):
        """Sample from the generator to create discriminator input."""
        # get generator token result
        sampled_tokens = (
            self.sample_from_softmax(gen_logits, use_softmax_sample)).detach()
        sampled_tokids = paddle.argmax(sampled_tokens, axis=-1)
        # update token only at mask position
        # gen_labels : [B, L], L contains -100(unmasked) or token value(masked)
        # mask_positions : [B, L], L contains 0(unmasked) or 1(masked)
        umask_positions = paddle.zeros_like(gen_labels)
        mask_positions = paddle.ones_like(gen_labels)
        mask_positions = paddle.where(gen_labels == -100, umask_positions,
                                      mask_positions)
        updated_inputs = self.update_inputs(inputs, sampled_tokids,
                                            mask_positions)
        # use inputs and updated_input to get discriminator labels
        labels = mask_positions * (paddle.ones_like(inputs) - paddle.equal(
            updated_inputs, raw_inputs).astype("int32"))
        return updated_inputs, labels, sampled_tokids

    def sample_from_softmax(self, logits, use_softmax_sample=True):
        if use_softmax_sample:
            # uniform_noise = paddle.uniform(logits.shape, dtype="float32", min=0, max=1)
            uniform_noise = paddle.rand(logits.shape, dtype=dtype_float)
            gumbel_noise = -paddle.log(-paddle.log(uniform_noise + 1e-9) + 1e-9)
        else:
            gumbel_noise = paddle.zeros_like(logits)
        # softmax_sample equal to sampled_tokids.unsqueeze(-1)
        softmax_sample = paddle.argmax(
            F.softmax(logits + gumbel_noise), axis=-1)
        # one hot
        return F.one_hot(softmax_sample, logits.shape[-1])

    def update_inputs(self, sequence, updates, positions):
        shape = sequence.shape
        assert len(
            shape
        ) == 2, "the dimension of inputs should be [batch_size, sequence_length]"
        B, L = shape
        N = positions.shape[1]
        assert N == L, "the dimension of inputs and mask should be same as [batch_size, sequence_length]"

        updated_sequence = ((paddle.ones_like(sequence) - positions) * sequence
                            ) + (positions * updates)

        return updated_sequence

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            attention_mask=None,
            raw_input_ids=None,
            gen_labels=None, ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`ConvBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`ConvBertModel`.
            position_ids (Tensor, optional):
                See :class:`ConvBertModel`.
            attention_mask (Tensor, optional):
                See :class:`ConvBertModel`.
            raw_input_ids(Tensor, optional):
                The raw input_ids. Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            gen_labels(Tensor, optional):
                The generator labels. Its data type should be `int64` and it has a shape of [batch_size, sequence_length].

        Returns:
            tuple: Returns tuple (``gen_logits``, ``disc_logits``, ``disc_labels``, ``attention_mask``).

            With the fields:

            - `gen_logits` (Tensor):
                a tensor of the generator prediction logits. Shape as `[batch_size, sequence_length, vocab_size]` and dtype as float32.

            - `disc_logits` (Tensor):
                a tensor of the discriminator prediction logits. Shape as `[batch_size, sequence_length]` and dtype as float32.

            - `disc_labels` (Tensor):
                a tensor of the discriminator prediction labels. Shape as `[batch_size, sequence_length]` and dtype as int64.

            - `attention_mask` (Tensor):
                See :class:`ConvBertModel`.
        """

        assert (gen_labels is not None), "gen_labels should not be None"

        gen_logits = self.generator(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        (
            disc_inputs,
            disc_labels,
            generator_predict_tokens, ) = self.get_discriminator_inputs(
                input_ids, raw_input_ids, gen_logits, gen_labels,
                self.use_softmax_sample)

        disc_logits = self.discriminator(disc_inputs, token_type_ids,
                                         position_ids, attention_mask)

        if attention_mask is None:
            attention_mask = (
                input_ids != self.discriminator.convbert.config["pad_token_id"])
        else:
            attention_mask = attention_mask.astype("bool")

        return gen_logits, disc_logits, disc_labels, attention_mask


class ConvBertPretrainingCriterion(nn.Layer):
    """

    Args:
        vocab_size(int):
            Vocabulary size of `inputs_ids` in `ConvBertModel`. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling `ConvBertModel`.
        gen_weight(float):
            This is the generator weight.
        disc_weight(float):
            This is the discriminator weight.

    """

    def __init__(self, vocab_size, gen_weight, disc_weight):
        super(ConvBertPretrainingCriterion, self).__init__()

        self.vocab_size = vocab_size
        self.gen_weight = gen_weight
        self.disc_weight = disc_weight
        self.gen_loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.disc_loss_fct = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
            self,
            generator_prediction_scores,
            discriminator_prediction_scores,
            generator_labels,
            discriminator_labels,
            attention_mask, ):
        # generator loss
        gen_loss = self.gen_loss_fct(
            paddle.reshape(generator_prediction_scores, [-1, self.vocab_size]),
            paddle.reshape(generator_labels, [-1]), )
        # todo: we can remove 4 lines after when CrossEntropyLoss(reduction='mean') improved
        umask_positions = paddle.zeros_like(generator_labels).astype(
            dtype_float)
        mask_positions = paddle.ones_like(generator_labels).astype(dtype_float)
        mask_positions = paddle.where(generator_labels == -100, umask_positions,
                                      mask_positions)
        if mask_positions.sum() == 0:
            gen_loss = paddle.to_tensor([0.0])
        else:
            gen_loss = gen_loss.sum() / mask_positions.sum()

        # discriminator loss
        seq_length = discriminator_labels.shape[1]
        disc_loss = self.disc_loss_fct(
            paddle.reshape(discriminator_prediction_scores, [-1, seq_length]),
            discriminator_labels.astype(dtype_float), )
        if attention_mask is not None:
            umask_positions = paddle.ones_like(discriminator_labels).astype(
                dtype_float)
            mask_positions = paddle.zeros_like(discriminator_labels).astype(
                dtype_float)
            use_disc_loss = paddle.where(attention_mask, disc_loss,
                                         mask_positions)
            umask_positions = paddle.where(attention_mask, umask_positions,
                                           mask_positions)
            disc_loss = use_disc_loss.sum() / umask_positions.sum()
        else:
            total_positions = paddle.ones_like(discriminator_labels).astype(
                dtype_float)
            disc_loss = disc_loss.sum() / total_positions.sum()

        return self.gen_weight * gen_loss + self.disc_weight * disc_loss


class ConvBertPooler(Layer):
    def __init__(self, hidden_size, pool_act="tanh"):
        super(ConvBertPooler, self).__init__()
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


class ConvBertForMultipleChoice(ConvBertPretrainedModel):
    """
    ConvBert Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like RocStories/SWAG tasks .

    Args:
        convbert (:class:`ConvBertModel`):
            An instance of ConvBertModel.
        num_choices (int, optional):
            The number of choices. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of ConvBert.
            If None, use the same value as `hidden_dropout_prob` of `ConvBertModel`
            instance `convbert`. Defaults to None.
    """

    def __init__(self, convbert, num_choices=2, dropout=None):
        super(ConvBertForMultipleChoice, self).__init__()
        self.num_choices = num_choices
        self.convbert = convbert
        self.pooler = ConvBertPooler(self.convbert.config["hidden_size"])
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.convbert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.convbert.config["hidden_size"], 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        The ConvBertForMultipleChoice forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ConvBertModel` and shape as [batch_size,num_choice, sequence_length].
            token_type_ids (Tensor, optional):
                See :class:`ConvBertModel` and shape as [batch_size,num_choice, sequence_length].
            position_ids(Tensor, optional):
                See :class:`ConvBertModel` and shape as [batch_size,num_choice, sequence_length].
            attention_mask (list, optional):
                See :class:`ConvBertModel` and shape as [batch_size,num_choice, sequence_length].

        Returns:
            Tensor: Returns tensor `reshaped_logits`, a tensor of the multiple choice classification logits.
            Shape as `[batch_size, num_choice]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ConvBertForMultipleChoice, ConvBertTokenizer

                tokenizer = ConvBertTokenizer.from_pretrained('convbert-base')
                model = ConvBertForMultipleChoice.from_pretrained('convbert-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
        """
        input_ids = input_ids.reshape(
            (-1, input_ids.shape[-1]))  # flat_input_ids: [bs*num_choice,seq_l]

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(
                (-1, token_type_ids.shape[-1]))
        if position_ids is not None:
            position_ids = position_ids.reshape((-1, position_ids.shape[-1]))
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(
                (-1, attention_mask.shape[-1]))

        sequence_output = self.convbert(input_ids, token_type_ids, position_ids,
                                        attention_mask)
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.reshape(
            (-1, self.num_choices))  # logits: (bs, num_choice)

        return reshaped_logits


class ConvBertForQuestionAnswering(ConvBertPretrainedModel):
    """
    ConvBert Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        convbert (:class:`ConvBertModel`):
            An instance of ConvBertModel.
            
    """

    def __init__(self, convbert):
        super(ConvBertForQuestionAnswering, self).__init__()
        self.convbert = convbert
        self.classifier = nn.Linear(self.convbert.config["hidden_size"], 2)
        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        The ConvBertForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ConvBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`ConvBertModel`.
            position_ids(Tensor, optional):
                See :class:`ConvBertModel`.
            attention_mask (list, optional):
                See :class:`ConvBertModel`.
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
                from paddlenlp.transformers import ConvBertForQuestionAnswering, ConvBertTokenizer

                tokenizer = ConvBertTokenizer.from_pretrained('convbert-base')
                model = ConvBertForQuestionAnswering.from_pretrained('convbert-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                start_logits = outputs[0]
                end_logits  = outputs[1]

        """
        sequence_output = self.convbert(
            input_ids,
            token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)
        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


ConvBertForPretraining = ConvBertForTotalPretraining
