# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from .. import PretrainedModel, register_base_model

__all__ = [
    'ErnieMModel', 'ErnieMPretrainedModel', 'ErnieMForSequenceClassification',
    'ErnieMForTokenClassification', 'ErnieMForQuestionAnswering'
]


class ErnieMEmbeddings(nn.Layer):
    r"""
    Include embeddings from word, position.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=514):
        super(ErnieMEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones
        position_ids += 2
        position_ids.stop_gradient = True
        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ErnieMPooler(nn.Layer):
    def __init__(self, hidden_size):
        super(ErnieMPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ErnieMPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained ERNIE-M models. It provides ERNIE-M related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models. 
    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "ernie-m-base": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 514,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "vocab_size": 250002,
            "pad_token_id": 1
        },
        "ernie-m-large": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "max_position_embeddings": 514,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "vocab_size": 250002,
            "pad_token_id": 1
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "ernie-m-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_m/ernie_m_base.pdparams",
            "ernie-m-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_m/ernie_m_large.pdparams",
        }
    }
    base_model_prefix = "ernie_m"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.ernie_m.config["initializer_range"],
                        shape=layer.weight.shape))


@register_base_model
class ErnieMModel(ErnieMPretrainedModel):
    r"""
    The bare ERNIE-M Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `ErnieMModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `ErnieMModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layers and pooler layer. Defaults to `768`.
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
            The vocabulary size of the `token_type_ids`.
            Defaults to `2`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer for initializing all weight matrices.
            Defaults to `0.02`.
            
            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`ErnieMPretrainedModel._init_weights()` for how weights are initialized in `ErnieMModel`.

        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `1`.

    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=514,
                 initializer_range=0.02,
                 pad_token_id=1):
        super(ErnieMModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = ErnieMEmbeddings(vocab_size, hidden_size,
                                           hidden_dropout_prob,
                                           max_position_embeddings)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            dim_feedforward=4 * hidden_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
            normalize_before=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = ErnieMPooler(hidden_size)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `[batch_size, num_tokens]` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Defaults to `None`, which means nothing needed to be prevented attention to.

        Returns:
            tuple: Returns tuple (``sequence_output``, ``pooled_output``).

            With the fields:

            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieMModel, ErnieMTokenizer

                tokenizer = ErnieMModel.from_pretrained('ernie-m-base')
                model = ErnieMTokenizer.from_pretrained('ernie-m-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                sequence_output, pooled_output = model(**inputs)

        """
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == 0).astype(self.pooler.dense.weight.dtype) * -1e4,
                axis=[1, 2])
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(
                attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
        attention_mask.stop_gradient = True
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class ErnieMForSequenceClassification(ErnieMPretrainedModel):
    r"""
    Ernie-M Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        ernie (ErnieMModel): 
            An instance of `paddlenlp.transformers.ErnieMModel`.
        num_classes (int, optional): 
            The number of classes. Default to `2`.
        dropout (float, optional): 
            The dropout probability for output of ERNIE-M. 
            If None, use the same value as `hidden_dropout_prob` 
            of `paddlenlp.transformers.ErnieMModel` instance. Defaults to `None`.
    """

    def __init__(self, ernie_m, num_classes=2, dropout=None):
        super(ErnieMForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie_m = ernie_m  # allow ernie_m to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie_m.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie_m.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieMModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieMModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieMModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieMModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieMForSequenceClassification, ErnieMTokenizer

                tokenizer = ErnieMTokenizer.from_pretrained('ernie-m-base')
                model = ErnieMForSequenceClassification.from_pretrained('ernie-m-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        _, pooled_output = self.ernie_m(
            input_ids, position_ids=position_ids, attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class ErnieMForQuestionAnswering(ErnieMPretrainedModel):
    """
    Ernie-M Model with a linear layer on top of the hidden-states
    output to compute `span_start_logits` and `span_end_logits`,
    designed for question-answering tasks like SQuAD.

    Args:
        ernie (`ErnieMModel`): 
            An instance of `ErnieMModel`.
    """

    def __init__(self, ernie_m):
        super(ErnieMForQuestionAnswering, self).__init__()
        self.ernie_m = ernie_m  # allow ernie_m to be config
        self.classifier = nn.Linear(self.ernie_m.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieMModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieMModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieMModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieMModel`.


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
                from paddlenlp.transformers import ErnieMForQuestionAnswering, ErnieMTokenizer

                tokenizer = ErnieMTokenizer.from_pretrained('ernie-m-base')
                model = ErnieMForQuestionAnswering.from_pretrained('ernie-m-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
        """

        sequence_output, _ = self.ernie_m(
            input_ids, position_ids=position_ids, attention_mask=attention_mask)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class ErnieMForTokenClassification(ErnieMPretrainedModel):
    r"""
    ERNIE-M Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        ernie (`ErnieMModel`): 
            An instance of `ErnieMModel`.
        num_classes (int, optional): 
            The number of classes. Defaults to `2`.
        dropout (float, optional): 
            The dropout probability for output of ERNIE-M. 
            If None, use the same value as `hidden_dropout_prob` 
            of `ErnieMModel` instance `ernie_m`. Defaults to `None`.
    """

    def __init__(self, ernie_m, num_classes=2, dropout=None):
        super(ErnieMForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie_m = ernie_m  # allow ernie_m to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie_m.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie_m.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieMModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieMModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieMModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieMModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieMForTokenClassification, ErnieMTokenizer

                tokenizer = ErnieMTokenizer.from_pretrained('ernie-m-base')
                model = ErnieMForTokenClassification.from_pretrained('ernie-m-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
        """
        sequence_output, _ = self.ernie_m(
            input_ids, position_ids=position_ids, attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits
