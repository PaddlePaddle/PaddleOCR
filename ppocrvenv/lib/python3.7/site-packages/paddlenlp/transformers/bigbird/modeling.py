# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 Google Research and The HuggingFace Inc. team.
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
from paddle.nn import Linear, Dropout, LayerNorm, LayerList, Layer
import paddle.nn.functional as F
import paddle.nn as nn

from ..attention_utils import _convert_param_attr_to_list, MultiHeadAttention, \
    AttentionRegistry
from .. import PretrainedModel, register_base_model

__all__ = [
    'BigBirdModel', 'BigBirdPretrainedModel', 'BigBirdForPretraining',
    'BigBirdPretrainingCriterion', 'BigBirdForSequenceClassification',
    'BigBirdPretrainingHeads', 'BigBirdForQuestionAnswering',
    'BigBirdForTokenClassification', 'BigBirdForMultipleChoice',
    'BigBirdForMaskedLM', 'BigBirdForCausalLM'
]


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


class TransformerEncoderLayer(Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None,
                 attention_type="bigbird",
                 block_size=1,
                 window_size=3,
                 num_global_blocks=1,
                 num_rand_blocks=1,
                 seed=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            attention_type=attention_type,
            block_size=block_size,
            window_size=window_size,
            num_global_blocks=num_global_blocks,
            num_rand_blocks=num_rand_blocks,
            seed=seed)
        self.linear1 = Linear(
            d_model, dim_feedforward, weight_attrs[1], bias_attr=bias_attrs[1])
        self.dropout = Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward, d_model, weight_attrs[1], bias_attr=bias_attrs[1])
        self.norm1 = LayerNorm(d_model, epsilon=1e-12)
        self.norm2 = LayerNorm(d_model, epsilon=1e-12)
        self.dropout1 = Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self.d_model = d_model

    def forward(self,
                src,
                src_mask=None,
                rand_mask_idx=None,
                query_mask=None,
                key_mask=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        src = self.self_attn(src, src, src, src_mask, rand_mask_idx, query_mask,
                             key_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(Layer):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = LayerList([(encoder_layer if i == 0 else
                                  type(encoder_layer)(**encoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = LayerNorm(self.layers[0].d_model, epsilon=1e-12)
        self.normalize_before = self.layers[0].normalize_before

    def forward(self,
                src,
                src_mask_list=None,
                rand_mask_idx_list=None,
                query_mask=None,
                key_mask=None):
        output = src
        if not self.normalize_before:
            output = self.norm(output)

        for i, mod in enumerate(self.layers):
            rand_mask_id = None
            if rand_mask_idx_list is not None:
                rand_mask_id = rand_mask_idx_list[i]
            if src_mask_list is None:
                output = mod(output, None, rand_mask_id, query_mask, key_mask)
            else:
                output = mod(output, src_mask_list[i], rand_mask_id, query_mask,
                             key_mask)

        if self.normalize_before:
            output = self.norm(output)
        return output


class BigBirdPooler(Layer):
    """
    Pool the result of BigBird Encoder
    """

    def __init__(self, hidden_size):
        super(BigBirdPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BigBirdEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 padding_idx=0):
        super(BigBirdEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class BigBirdPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained BigBird models. It provides BigBird related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "bigbird-base-uncased": {
            "num_layers": 12,
            "vocab_size": 50358,
            "nhead": 12,
            "attn_dropout": 0.1,
            "dim_feedforward": 3072,
            "activation": "gelu",
            "normalize_before": False,
            "block_size": 16,
            "window_size": 3,
            "num_global_blocks": 2,
            "num_rand_blocks": 3,
            "seed": None,
            "pad_token_id": 0,
            "hidden_size": 768,
            "hidden_dropout_prob": 0.1,
            "max_position_embeddings": 4096,
            "type_vocab_size": 2,
            "num_labels": 2,
            "initializer_range": 0.02,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "bigbird-base-uncased":
            "https://bj.bcebos.com/paddlenlp/models/transformers/bigbird/bigbird-base-uncased.pdparams",
        }
    }
    base_model_prefix = "bigbird"

    def init_weights(self, layer):
        # Initialization hook
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.bigbird.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class BigBirdModel(BigBirdPretrainedModel):
    """
    The bare BigBird Model outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        num_layers (int):
            Number of hidden layers in the Transformer encoder.
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `BigBirdModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `BigBirdModel`.
        nhead (int):
            Number of attention heads for each attention layer in the Transformer encoder.
        attn_dropout (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        dim_feedforward (int, optional):
            Dimensionality of the feed-forward (ff) layer in the Transformer encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `3072`.
        activation (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"``, ``"silu"`` and ``"gelu_new"`` are supported.
            Defaults to `"gelu"`.
        normalize_before (bool, optional):
            Indicates whether to put layer normalization into preprocessing of MHA and FFN sub-layers.
            If True, pre-process is layer normalization and post-precess includes dropout,
            residual connection. Otherwise, no pre-process and post-precess includes dropout,
            residual connection, layer normalization.
            Defaults to `False`.
        block_size (int, optional):
            The block size for the attention mask.
            Defaults to `1`.
        window_size (int, optional):
            The number of block in a window.
            Defaults to `3`.
        num_global_blocks (int, optional):
            Number of global blocks per sequence.
            Defaults to `1`.
        num_rand_blocks (int, optional):
            Number of random blocks per row.
            Defaults to `2`.
        seed (int, optional):
            The random seed for generating random block id.
            Defaults to ``None``.
        pad_token_id (int, optional):
            The index of padding token for BigBird embedding.
            Defaults to ``0``.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layer and pooler layer.
            Defaults to `768`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids`.
            Defaults to `2`.
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 nhead,
                 attn_dropout=0.1,
                 dim_feedforward=3072,
                 activation="gelu",
                 normalize_before=False,
                 block_size=1,
                 window_size=3,
                 num_global_blocks=1,
                 num_rand_blocks=2,
                 seed=None,
                 pad_token_id=0,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 **kwargs):
        super(BigBirdModel, self).__init__()
        # embedding
        self.embeddings = BigBirdEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size, pad_token_id)

        # encoder
        encoder_layer = TransformerEncoderLayer(
            hidden_size,
            nhead,
            dim_feedforward,
            attn_dropout,
            activation,
            normalize_before=normalize_before,
            attention_type="bigbird",
            block_size=block_size,
            window_size=window_size,
            num_global_blocks=num_global_blocks,
            num_rand_blocks=num_rand_blocks,
            seed=seed)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        # pooler
        self.pooler = BigBirdPooler(hidden_size)
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers

    def _process_mask(self, input_ids, attention_mask_list=None):
        # [B, T]
        attention_mask = (input_ids == self.pad_token_id
                          ).astype(self.pooler.dense.weight.dtype)
        # [B, 1, T, 1]
        query_mask = paddle.unsqueeze(attention_mask, axis=[1, 3])
        # [B, 1, 1, T]
        key_mask = paddle.unsqueeze(attention_mask, axis=[1, 2])
        query_mask = 1 - query_mask
        key_mask = 1 - key_mask
        return attention_mask_list, query_mask, key_mask

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask_list=None,
                rand_mask_idx_list=None):
        r"""
        The BigBirdModel forward method, overrides the __call__() special method.

        Args:
            input_ids (`Tensor`):
                Indices of input sequence tokens in the vocabulary.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            token_type_ids (`Tensor`, optional):
                Segment token indices to indicate first and second portions of the inputs.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to ``None``, which means we don't add segment embeddings.
            attention_mask_list (list, optional):
                A list which contains some tensors used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Defaults to `None`, which means nothing needed to be prevented attention to.
            rand_mask_idx_list (`list`, optional):
                A list which contains some tensors used in bigbird random block.

        Returns:
            tuple: Returns tuple (`encoder_output`, `pooled_output`).

            With the fields:

            - encoder_output (Tensor):
                Sequence of output at the last layer of the model.
                Its data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].

            - pooled_output (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].

        Examples:
            .. code-block::

                import paddle
                from paddlenlp.transformers import BigBirdModel, BigBirdTokenizer
                from paddlenlp.transformers import create_bigbird_rand_mask_idx_list

                tokenizer = BigBirdTokenizer.from_pretrained('bigbird-base-uncased')
                model = BigBirdModel.from_pretrained('bigbird-base-uncased')
                config = model.config
                max_seq_len = 512
                input_ids = tokenizer.convert_tokens_to_ids(
                    tokenizer(
                        "This is a docudrama story on the Lindy Chamberlain case and a look at "
                        "its impact on Australian society It especially looks at the problem of "
                        "innuendo gossip and expectation when dealing with reallife dramasbr br "
                        "One issue the story deals with is the way it is expected people will all "
                        "give the same emotional response to similar situations Not everyone goes "
                        "into wild melodramatic hysterics to every major crisis Just because the "
                        "characters in the movies and on TV act in a certain way is no reason to "
                        "expect real people to do so"
                    ))
                input_ids.extend([0] * (max_seq_len - len(input_ids)))
                seq_len = len(input_ids)
                input_ids = paddle.to_tensor([input_ids])
                rand_mask_idx_list = create_bigbird_rand_mask_idx_list(
                    config["num_layers"], seq_len, seq_len, config["nhead"],
                    config["block_size"], config["window_size"], config["num_global_blocks"],
                    config["num_rand_blocks"], config["seed"])
                rand_mask_idx_list = [
                    paddle.to_tensor(rand_mask_idx) for rand_mask_idx in rand_mask_idx_list
                ]
                output = model(input_ids, rand_mask_idx_list=rand_mask_idx_list)
        """
        embedding_output = self.embeddings(input_ids, token_type_ids)
        attention_mask_list, query_mask, key_mask = self._process_mask(
            input_ids, attention_mask_list)
        encoder_output = self.encoder(embedding_output, attention_mask_list,
                                      rand_mask_idx_list, query_mask, key_mask)
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output


class BigBirdForSequenceClassification(BigBirdPretrainedModel):
    """
    BigBird Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        bigbird (:class:`BigBirdModel`):
            An instance of :class:`BigBirdModel`.
        num_classes (int, optional):
            The number of classes. Defaults to `None`.
    """

    def __init__(self, bigbird, num_classes=None):
        super(BigBirdForSequenceClassification, self).__init__()
        self.bigbird = bigbird
        if num_classes is None:
            num_classes = self.bigbird.config["num_labels"]
        self.linear = nn.Linear(self.bigbird.config["hidden_size"], num_classes)
        self.dropout = nn.Dropout(
            self.bigbird.config['hidden_dropout_prob'], mode="upscale_in_train")
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask_list=None,
                rand_mask_idx_list=None):
        r"""
        The BigBirdForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`BigBirdModel`.
            token_type_ids (Tensor):
                See :class:`BigBirdModel`.
            attention_mask_list (list):
                See :class:`BigBirdModel`.
            rand_mask_idx_list (list):
                See :class:`BigBirdModel`.

        Returns:
            Tensor: Returns tensor `output`, a tensor of the input text classification logits.
            Its data type should be float32 and it has a shape of [batch_size, num_classes].

        Examples:
            .. code-block::

                import paddle
                from paddlenlp.transformers import BigBirdForSequenceClassification, BigBirdTokenizer
                from paddlenlp.transformers import create_bigbird_rand_mask_idx_list

                tokenizer = BigBirdTokenizer.from_pretrained('bigbird-base-uncased')
                model = BigBirdForSequenceClassification.from_pretrained('bigbird-base-uncased')
                config = model.bigbird.config
                max_seq_len = 512
                input_ids = tokenizer.convert_tokens_to_ids(
                    tokenizer(
                        "This is a docudrama story on the Lindy Chamberlain case and a look at "
                        "its impact on Australian society It especially looks at the problem of "
                        "innuendo gossip and expectation when dealing with reallife dramasbr br "
                        "One issue the story deals with is the way it is expected people will all "
                        "give the same emotional response to similar situations Not everyone goes "
                        "into wild melodramatic hysterics to every major crisis Just because the "
                        "characters in the movies and on TV act in a certain way is no reason to "
                        "expect real people to do so"
                    ))
                input_ids.extend([0] * (max_seq_len - len(input_ids)))
                seq_len = len(input_ids)
                input_ids = paddle.to_tensor([input_ids])
                rand_mask_idx_list = create_bigbird_rand_mask_idx_list(
                    config["num_layers"], seq_len, seq_len, config["nhead"],
                    config["block_size"], config["window_size"], config["num_global_blocks"],
                    config["num_rand_blocks"], config["seed"])
                rand_mask_idx_list = [
                    paddle.to_tensor(rand_mask_idx) for rand_mask_idx in rand_mask_idx_list
                ]
                output = model(input_ids, rand_mask_idx_list=rand_mask_idx_list)
                print(output)
        """
        _, pooled_output = self.bigbird(
            input_ids,
            token_type_ids,
            attention_mask_list=attention_mask_list,
            rand_mask_idx_list=rand_mask_idx_list)
        output = self.dropout(pooled_output)
        output = self.linear(output)
        return output


class BigBirdLMPredictionHead(Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(BigBirdLMPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=1e-12)
        self.decoder_weight = self.create_parameter(
            shape=[vocab_size, hidden_size],
            dtype=self.transform.weight.dtype,
            is_bias=False) if embedding_weights is None else embedding_weights
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
        hidden_states = paddle.tensor.matmul(
            hidden_states, self.decoder_weight,
            transpose_y=True) + self.decoder_bias
        return hidden_states


class BigBirdPretrainingHeads(Layer):
    """
    The BigBird pretraining heads for a pretraining task.

    Args:
        hidden_size (int):
            See :class:`BigBirdModel`.
        vocab_size (int):
            See :class:`BigBirdModel`.
        activation (str):
            See :class:`BigBirdModel`.
        embedding_weights (Tensor, optional):
            The weight of pretraining embedding layer. Its data type should be float32
            and its shape is [hidden_size, vocab_size].
            If set to `None`, use normal distribution to initialize weight.
            Defaults to `None`.
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(BigBirdPretrainingHeads, self).__init__()
        self.predictions = BigBirdLMPredictionHead(
            hidden_size, vocab_size, activation, embedding_weights)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        r"""
        The BigBirdPretrainingHeads forward method, overrides the __call__() special method.

        Args:
            sequence_output (Tensor):
                The sequence output of BigBirdModel. Its data type should be float32 and
                has a shape of [batch_size, sequence_length, hidden_size].
            pooled_output (Tensor):
                The pooled output of BigBirdModel. Its data type should be float32 and
                has a shape of [batch_size, hidden_size].
            masked_positions (Tensor):
                A tensor indicates positions to be masked in the position embedding.
                Its data type should be int64 and its shape is [batch_size, mask_token_num].
                `mask_token_num` is the number of masked tokens. It should be no bigger than `sequence_length`.
                Defaults to `None`, which means we output hidden-states of all tokens in masked token prediction.

        Returns:
            tuple: (``prediction_scores``, ``seq_relationship_score``).

            With the fields:

            - prediction_scores (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size].

            - seq_relationship_score (Tensor):
                The logits whether 2 sequences are NSP relationship. Its data type should be float32 and
                has a shape of [batch_size, 2].
        """
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BigBirdForPretraining(BigBirdPretrainedModel):
    """
    BigBird Model with pretraining tasks on top.

    Args:
        bigbird (:class:`BigBirdModel`):
            An instance of :class:`BigBirdModel`.

    """

    def __init__(self, bigbird):
        super(BigBirdForPretraining, self).__init__()
        self.bigbird = bigbird
        self.cls = BigBirdPretrainingHeads(
            self.bigbird.config["hidden_size"],
            self.bigbird.config["vocab_size"],
            self.bigbird.config["activation"],
            embedding_weights=self.bigbird.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                rand_mask_idx_list=None,
                masked_positions=None):
        r"""
        The BigBirdForPretraining forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`BigBirdModel`.
            token_type_ids (Tensor):
                See :class:`BigBirdModel`.
            attention_mask_list (list):
                See :class:`BigBirdModel`.
            rand_mask_idx_list (list):
                See :class:`BigBirdModel`.
            masked_positions (list):
                A tensor indicates positions to be masked in the position embedding.
                Its data type should be int64 and its shape is [batch_size, mask_token_num].
                `mask_token_num` is the number of masked tokens. It should be no bigger than `sequence_length`.
                Defaults to `None`, which means we output hidden-states of all tokens in masked token prediction.

        Returns:
            tuple: Returns tuple (`prediction_scores`, `seq_relationship_score`).

            With the fields:

            - prediction_scores (Tensor):
                The scores of masked token prediction.
                Its data type should be float32 and its shape is [batch_size, sequence_length, vocab_size].

            - seq_relationship_score (Tensor):
                The scores of next sentence prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

        Examples:
            .. code-block::

                import paddle
                from paddlenlp.transformers import BigBirdForPretraining, BigBirdTokenizer
                from paddlenlp.transformers import create_bigbird_rand_mask_idx_list

                tokenizer = BigBirdTokenizer.from_pretrained('bigbird-base-uncased')
                model = BigBirdForPretraining.from_pretrained('bigbird-base-uncased')
                config = model.bigbird.config
                max_seq_len = 512
                input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = tokenizer.encode(
                        "This is a docudrama story on the Lindy Chamberlain case and a look at "
                        "its impact on Australian society It especially looks at the problem of "
                        "innuendo gossip and expectation when dealing with reallife dramasbr br "
                        "One issue the story deals with is the way it is expected people will all "
                        "give the same emotional response to similar situations Not everyone goes "
                        "into wild melodramatic hysterics to every major crisis Just because the "
                        "characters in the movies and on TV act in a certain way is no reason to "
                        "expect real people to do so", max_seq_len=max_seq_len)

                seq_len = len(input_ids)
                input_ids = paddle.to_tensor([input_ids])
                rand_mask_idx_list = create_bigbird_rand_mask_idx_list(
                    config["num_layers"], seq_len, seq_len, config["nhead"],
                    config["block_size"], config["window_size"], config["num_global_blocks"],
                    config["num_rand_blocks"], config["seed"])
                rand_mask_idx_list = [
                    paddle.to_tensor(rand_mask_idx) for rand_mask_idx in rand_mask_idx_list
                ]
                output = model(input_ids, rand_mask_idx_list=rand_mask_idx_list)
                print(output)
        """
        outputs = self.bigbird(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask_list=None,
            rand_mask_idx_list=rand_mask_idx_list)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output, masked_positions)
        return prediction_scores, seq_relationship_score


class BigBirdPretrainingCriterion(paddle.nn.Layer):
    """
    BigBird Criterion for a pretraining task on top.

    Args:
        vocab_size (int):
            See :class:`BigBirdModel`.
        use_nsp (bool, optional):
            It decides whether it considers Next Sentence Prediction loss.
            Defaults to `False`.
        ignore_index (int):
            Specifies a target value that is ignored and does
            not contribute to the input gradient. Only valid
            if :attr:`soft_label` is set to :attr:`False`.
            Defaults to `0`.
    """

    def __init__(self, vocab_size, use_nsp=False, ignore_index=0):
        super(BigBirdPretrainingCriterion, self).__init__()
        # CrossEntropyLoss is expensive since the inner reshape (copy)
        self.loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size
        self.use_nsp = use_nsp
        self.ignore_index = ignore_index

    def forward(self, prediction_scores, seq_relationship_score,
                masked_lm_labels, next_sentence_labels, masked_lm_scale,
                masked_lm_weights):
        r"""
        The BigBirdPretrainingCriterion forward method, overrides the __call__() special method.

        Args:
            prediction_scores (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size].
            seq_relationship_score (Tensor):
                The scores of next sentence prediction.
                Its data type should be float32 and its shape is [batch_size, 2].
            masked_lm_labels (Tensor):
                The labels of the masked language modeling, its dimensionality is equal to `prediction_scores`.
                Its data type should be int64. If `masked_positions` is None, its shape is [batch_size, sequence_length, 1].
                Otherwise, its shape is [batch_size, mask_token_num, 1].
            next_sentence_labels (Tensor):
                The labels of the next sentence prediction task, the dimensionality of `next_sentence_labels`
                is equal to `seq_relation_labels`. Its data type should be int64 and its shape is [batch_size, 1].
            masked_lm_scale (Tensor or int):
                The scale of masked tokens. Used for the normalization of masked language modeling loss.
                If it is a `Tensor`, its data type should be int64 and its shape is equal to `prediction_scores`.
            masked_lm_weights (Tensor):
                The weight of masked tokens. Its data type should be float32 and its shape
                is [mask_token_num, 1].

        Returns:
            Tensor: The pretraining loss, equals to the sum of `masked_lm_loss` plus the mean of `next_sentence_loss`.
            Its data type should be float32 and its shape is [1].

        Example:
            .. code-block::

                import numpy as np
                import paddle
                from paddlenlp.transformers import BigBirdForPretraining, BigBirdTokenizer, BigBirdPretrainingCriterion
                from paddlenlp.transformers import create_bigbird_rand_mask_idx_list

                tokenizer = BigBirdTokenizer.from_pretrained('bigbird-base-uncased')
                model = BigBirdForPretraining.from_pretrained('bigbird-base-uncased')
                config = model.bigbird.config
                criterion = BigBirdPretrainingCriterion(config["vocab_size"], False)
                max_seq_len = 512
                max_pred_length=75
                input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = tokenizer.encode(
                        "This is a docudrama story on the Lindy Chamberlain case and a look at "
                        "its impact on Australian society It especially looks at the problem of "
                        "innuendo gossip and expectation when dealing with reallife dramasbr br "
                        "One issue the story deals with is the way it is expected people will all "
                        "give the same emotional response to similar situations Not everyone goes "
                        "into wild melodramatic hysterics to every major crisis Just because the "
                        "characters in the movies and on TV act in a certain way is no reason to "
                        "expect real people to do so", max_seq_len=max_seq_len, max_pred_len=max_pred_length)

                seq_len = len(input_ids)
                masked_lm_positions_tmp = np.full(seq_len, 0, dtype=np.int32)
                masked_lm_ids_tmp = np.full([seq_len, 1], -1, dtype=np.int64)
                masked_lm_weights_tmp = np.full([seq_len], 0, dtype="float32")

                mask_token_num = 0
                for i, x in enumerate([input_ids]):
                    for j, pos in enumerate(masked_lm_positions):
                        masked_lm_positions_tmp[mask_token_num] = i * seq_len + pos
                        masked_lm_ids_tmp[mask_token_num] = masked_lm_ids[j]
                        masked_lm_weights_tmp[mask_token_num] = masked_lm_weights[j]

                masked_lm_positions = masked_lm_positions_tmp
                masked_lm_ids = masked_lm_ids_tmp
                masked_lm_weights = masked_lm_weights_tmp
                print(masked_lm_ids.shape)
                input_ids = paddle.to_tensor([input_ids])
                masked_lm_positions = paddle.to_tensor(masked_lm_positions)
                masked_lm_ids = paddle.to_tensor(masked_lm_ids, dtype='int64')
                masked_lm_weights = paddle.to_tensor(masked_lm_weights)
                masked_lm_scale = 1.0
                next_sentence_labels = paddle.zeros(shape=(1, 1), dtype='int64')

                rand_mask_idx_list = create_bigbird_rand_mask_idx_list(
                    config["num_layers"], seq_len, seq_len, config["nhead"],
                    config["block_size"], config["window_size"], config["num_global_blocks"],
                    config["num_rand_blocks"], config["seed"])
                rand_mask_idx_list = [
                    paddle.to_tensor(rand_mask_idx) for rand_mask_idx in rand_mask_idx_list
                ]
                prediction_scores, seq_relationship_score = model(input_ids, rand_mask_idx_list=rand_mask_idx_list, masked_positions=masked_lm_positions)

                loss = criterion(prediction_scores, seq_relationship_score,
                                masked_lm_ids, next_sentence_labels,
                                masked_lm_scale, masked_lm_weights)
                print(loss)
        """
        masked_lm_loss = F.cross_entropy(
            prediction_scores,
            masked_lm_labels,
            ignore_index=self.ignore_index,
            reduction='none')
        masked_lm_loss = paddle.transpose(masked_lm_loss, [1, 0])
        masked_lm_loss = paddle.sum(masked_lm_loss * masked_lm_weights) / (
            paddle.sum(masked_lm_weights) + 1e-5)
        scale = 1.0
        if not self.use_nsp:
            scale = 0.0
        next_sentence_loss = F.cross_entropy(
            seq_relationship_score, next_sentence_labels, reduction='none')
        return masked_lm_loss + paddle.mean(next_sentence_loss) * scale


class BigBirdIntermediate(Layer):
    def __init__(self, hidden_size, dim_feedforward, activation):
        super().__init__()
        self.dense = nn.Linear(hidden_size, dim_feedforward)
        if isinstance(activation, str):
            self.intermediate_act_fn = ACT2FN[activation]
        else:
            self.intermediate_act_fn = activation

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BigBirdOutput(Layer):
    def __init__(self, hidden_size, dim_feedforward, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(dim_feedforward, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BigBirdForQuestionAnswering(BigBirdPretrainedModel):
    """
    BigBird Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        bigbird (:class:`BigBirdModel`):
            An instance of BigBirdModel.
        dropout (float, optional):
            The dropout probability for output of BigBirdModel.
            If None, use the same value as `hidden_dropout_prob` of `BigBirdModel`
            instance `bigbird`. Defaults to `None`.
        """

    def __init__(self, bigbird, dropout=None):
        super(BigBirdForQuestionAnswering, self).__init__()
        self.bigbird = bigbird  # allow bigbird to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.bigbird.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bigbird.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask_list=None,
                rand_mask_idx_list=None):
        r"""
        The BigBirdForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`BigBirdModel`.
            token_type_ids (Tensor, optional):
                See :class:`BigBirdModel`.
            attention_mask_list (`List`):
                See :class:`BigBirdModel`.
            rand_mask_idx_list (`List`):
                See :class:`BigBirdModel`.

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
                from paddlenlp.transformers.bigbird.modeling import BigBirdForQuestionAnswering
                from paddlenlp.transformers.bigbird.tokenizer import BigBirdTokenizer

                tokenizer = BigBirdTokenizer.from_pretrained('bigbird-base-uncased')
                model = BigBirdForQuestionAnswering.from_pretrained('bigbird-base-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                start_logits = outputs[0]
                end_logits  =outputs[1]
        """
        sequence_output, _ = self.bigbird(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask_list=attention_mask_list,
            rand_mask_idx_list=rand_mask_idx_list)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits

    @staticmethod
    def prepare_question_mask(q_lengths, maxlen):
        mask = paddle.arange(0, maxlen).unsqueeze_(0)
        mask = mask < q_lengths
        return mask


class BigBirdForTokenClassification(BigBirdPretrainedModel):
    """
    BigBird Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        bigbird (:class:`BigBirdModel`):
            An instance of BigBirdModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of BIGBIRD.
            If None, use the same value as `hidden_dropout_prob` of `BigBirdModel`
            instance `bigbird`. Defaults to None.
    """

    def __init__(self, bigbird, num_classes=2, dropout=None):
        super(BigBirdForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.bigbird = bigbird  # allow bigbird to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.bigbird.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bigbird.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask_list=None,
                rand_mask_idx_list=None):
        r"""
        The BigBirdForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`BigBirdModel`.
            token_type_ids (Tensor, optional):
                See :class:`BigBirdModel`.
            attention_mask_list (`List`):
                See :class:`BigBirdModel`.
            rand_mask_idx_list (`List`):
                See :class:`BigBirdModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.bigbird.modeling import BigBirdForTokenClassification
                from paddlenlp.transformers.bigbird.tokenizer import BigBirdTokenizer

                tokenizer = BigBirdTokenizer.from_pretrained('bigbird-base-uncased')
                model = BigBirdForTokenClassification.from_pretrained('bigbird-base-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs
        """
        sequence_output, _ = self.bigbird(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask_list=attention_mask_list,
            rand_mask_idx_list=rand_mask_idx_list)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class BigBirdForMultipleChoice(BigBirdPretrainedModel):
    """
    BigBird Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like RocStories/SWAG tasks .

    Args:
        bigbird (:class:`BigBirdModel`):
            An instance of BigBirdModel.
        num_choices (int, optional):
            The number of choices. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of BIGBIRD.
            If None, use the same value as `hidden_dropout_prob` of `BigBirdModel`
            instance `bigbird`. Defaults to None.
    """

    def __init__(self, bigbird, num_choices=2, dropout=None):
        super(BigBirdForMultipleChoice, self).__init__()
        self.bigbird = bigbird  # allow bigbird to be config
        self.num_choices = num_choices
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.bigbird.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bigbird.config["hidden_size"], 1)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                attention_mask_list=None,
                rand_mask_idx_list=None):
        r"""
        The BigBirdForMultipleChoice forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`BigBirdModel` and shape as [batch_size, num_choice, sequence_length].
            attention_mask_list (`List`):
                See :class:`BigBirdModel`  and shape as [batch_size, num_choice, n_head, sequence_length, sequence_length].
            rand_mask_idx_list (`List`):
                See :class:`BigBirdModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, 1]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.bigbird.modeling import BigBirdForMultipleChoice
                from paddlenlp.transformers.bigbird.tokenizer import BigBirdTokenizer

                tokenizer = BigBirdTokenizer.from_pretrained('bigbird-base-uncased')
                model = BigBirdForTokenClassification.from_pretrained('bigbird-base-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs
        """
        # input_ids: [bs, num_choice, seq_l]
        input_ids = input_ids.reshape(shape=(
            -1, input_ids.shape[-1]))  # flat_input_ids: [bs*num_choice,seq_l]

        if attention_mask_list is not None:
            attention_mask_list = attention_mask_list.reshape(shape=(
                -1, *attention_mask_list.shape[2:]))

        if rand_mask_idx_list is not None:
            rand_mask_idx_list = rand_mask_idx_list.reshape(shape=(
                -1, *rand_mask_idx_list.shape[2:]))

        _, pooled_output = self.bigbird(
            input_ids,
            attention_mask_list=attention_mask_list,
            rand_mask_idx_list=rand_mask_idx_list)

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.reshape(
            shape=(-1, self.num_choices))  # logits: (bs, num_choice)

        return reshaped_logits


class BigBirdForMaskedLM(BigBirdPretrainedModel):
    """
    BigBird Model with a `language modeling` head on top.

    Args:
        BigBird (:class:`BigBirdModel`):
            An instance of :class:`BigBirdModel`.

    """

    def __init__(self, bigbird):
        super(BigBirdForMaskedLM, self).__init__()
        self.bigbird = bigbird
        self.lm_head = BigBirdLMPredictionHead(
            self.bigbird.config["hidden_size"],
            self.bigbird.config["vocab_size"],
            self.bigbird.config["activation"],
            self.bigbird.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                attention_mask_list=None,
                rand_mask_idx_list=None,
                labels=None):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`BigBirdModel`.
            attention_mask_list (`List`):
                See :class:`BigBirdModel`.
            rand_mask_idx_list (`List`):
                See :class:`BigBirdModel`.
            labels (Tensor, optional):
                The Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ..., vocab_size]`` Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels in ``[0, ..., vocab_size]`` Its shape is [batch_size, sequence_length].

        Returns:
            tuple: Returns tuple (`masked_lm_loss`, `prediction_scores`, ``sequence_output`).

            With the fields:

            - `masked_lm_loss` (Tensor):
                The masked lm loss. Its data type should be float32 and its shape is [1].

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32. Its shape is [batch_size, sequence_length, vocab_size].

            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model. Its data type should be float32. Its shape is `[batch_size, sequence_length, hidden_size]`.


        """
        sequence_output, _ = self.bigbird(
            input_ids,
            attention_mask_list=attention_mask_list,
            rand_mask_idx_list=rand_mask_idx_list)
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.reshape(shape=(
                    -1, self.bigbird.config["vocab_size"])),
                labels.reshape(shape=(-1, )), )
            return masked_lm_loss, prediction_scores, sequence_output

        return prediction_scores, sequence_output


class BigBirdForCausalLM(BigBirdPretrainedModel):
    """
    BigBird Model for casual language model tasks.

    Args:
        BigBird (:class:`BigBirdModel`):
            An instance of :class:`BigBirdModel`.

    """

    def __init__(self, bigbird):
        super(BigBirdForCausalLM, self).__init__()
        self.bigbird = bigbird
        self.lm_head = BigBirdLMPredictionHead(
            self.bigbird.config["hidden_size"],
            self.bigbird.config["vocab_size"],
            self.bigbird.config["activation"],
            self.bigbird.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                attention_mask_list=None,
                rand_mask_idx_list=None,
                labels=None):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`BigBirdModel`.
            attention_mask_list (`List`):
                See :class:`BigBirdModel`.
            rand_mask_idx_list (`List`):
                See :class:`BigBirdModel`.
            labels (Tensor, optional):
                The Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ..., vocab_size]`` Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels in ``[0, ..., vocab_size]`` Its shape is [batch_size, sequence_length].

        Returns:
            tuple: Returns tuple (`masked_lm_loss`, `prediction_scores`, ``sequence_output`).

            With the fields:

            - `masked_lm_loss` (Tensor):
                The masked lm loss. Its data type should be float32 and its shape is [1].

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32. Its shape is [batch_size, sequence_length, vocab_size].

            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model. Its data type should be float32. Its shape is `[batch_size, sequence_length, hidden_size]`.


        """
        sequence_output, _ = self.bigbird(
            input_ids,
            attention_mask_list=attention_mask_list,
            rand_mask_idx_list=rand_mask_idx_list)
        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :]
            labels = labels[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                paddle.reshape(shifted_prediction_scores,
                               [-1, self.bigbird.config['vocab_size']]),
                paddle.reshape(labels, [-1]))

            return lm_loss, prediction_scores, sequence_output

        return prediction_scores, sequence_output
