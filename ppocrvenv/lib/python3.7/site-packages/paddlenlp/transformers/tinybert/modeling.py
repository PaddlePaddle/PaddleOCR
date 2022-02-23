# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 Huawei Technologies Co., Ltd.
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

from ..bert.modeling import BertPooler, BertEmbeddings
from .. import PretrainedModel, register_base_model

__all__ = [
    'TinyBertModel',
    'TinyBertPretrainedModel',
    'TinyBertForPretraining',
    'TinyBertForSequenceClassification',
]


class TinyBertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained TinyBERT models. It provides TinyBERT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading
    and loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "tinybert-4l-312d": {
            "vocab_size": 30522,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-4l-312d-v2": {
            "vocab_size": 30522,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d-v2": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-4l-312d-zh": {
            "vocab_size": 21128,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d-zh": {
            "vocab_size": 21128,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "tinybert-4l-312d":
            "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-4l-312d.pdparams",
            "tinybert-6l-768d":
            "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-6l-768d.pdparams",
            "tinybert-4l-312d-v2":
            "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-4l-312d-v2.pdparams",
            "tinybert-6l-768d-v2":
            "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-6l-768d-v2.pdparams",
            "tinybert-4l-312d-zh":
            "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-4l-312d-zh.pdparams",
            "tinybert-6l-768d-zh":
            "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-6l-768d-zh.pdparams",
        }
    }
    base_model_prefix = "tinybert"

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
                        self.tinybert.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class TinyBertModel(TinyBertPretrainedModel):
    """
    The bare TinyBert Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `TinyBertModel`. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling `TinyBertModel`.
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
            The maximum value of the dimensionality of position encoding. The dimensionality of position encoding
            is the dimensionality of the sequence in `TinyBertModel`.
            Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids` passed when calling `~ transformers.TinyBertModel`.
            Defaults to `16`.

        initializer_range (float, optional):
            The standard deviation of the normal initializer.
            Defaults to `0.02`.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`TinyBertPretrainedModel.init_weights()` for how weights are initialized in `TinyBertModel`.

        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        fit_size (int, optional):
            Dimensionality of the output layer of `fit_dense(s)`, which is the hidden size of the teacher model.
            `fit_dense(s)` means a hidden states' transformation from student to teacher.
            `fit_dense(s)` will be generated when bert model is distilled during the training, and will not be generated
            during the prediction process.
            `fit_denses` is used in v2 models and it has `num_hidden_layers+1` layers.
            `fit_dense` is used in other pretraining models and it has one linear layer.
            Defaults to `768`.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0,
                 fit_size=768):
        super(TinyBertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = BertPooler(hidden_size)
        # fit_dense(s) means a hidden states' transformation from student to teacher.
        # `fit_denses` is used in v2 model, and `fit_dense` is used in other pretraining models.
        self.fit_denses = nn.LayerList([
            nn.Linear(hidden_size, fit_size)
            for i in range(num_hidden_layers + 1)
        ])
        self.fit_dense = nn.Linear(hidden_size, fit_size)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        r'''
        The TinyBertModel forward method, overrides the `__call__()` special method.

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
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention to some unwanted positions,
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
            tuple: Returns tuple (`encoder_output`, `pooled_output`).

            With the fields:

            - `encoder_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import TinyBertModel, TinyBertTokenizer

                tokenizer = TinyBertTokenizer.from_pretrained('tinybert-4l-312d')
                model = TinyBertModel.from_pretrained('tinybert-4l-312d')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP! ")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        '''

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e4,
                axis=[1, 2])
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layer = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(encoded_layer)

        return encoded_layer, pooled_output


class TinyBertForPretraining(TinyBertPretrainedModel):
    """
    TinyBert Model with pretraining tasks on top.

    Args:
        tinybert (:class:`TinyBertModel`):
            An instance of :class:`TinyBertModel`.

    """

    def __init__(self, tinybert):
        super(TinyBertForPretraining, self).__init__()
        self.tinybert = tinybert
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        r"""
        The TinyBertForPretraining forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`TinyBertModel`.
            token_tycpe_ids (Tensor, optional):
                See :class:`TinyBertModel`.
            attention_mask (Tensor, optional):
                See :class:`TinyBertModel`.

        Returns:
            Tensor: Returns tensor `sequence_output`, sequence of hidden-states at the last layer of the model.
            It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.tinybert.modeling import TinyBertForPretraining
                from paddlenlp.transformers.tinybert.tokenizer import TinyBertTokenizer

                tokenizer = TinyBertTokenizer.from_pretrained('tinybert-4l-312d')
                model = TinyBertForPretraining.from_pretrained('tinybert-4l-312d')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP! ")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]


        """
        sequence_output, pooled_output = self.tinybert(
            input_ids, token_type_ids, attention_mask)

        return sequence_output


class TinyBertForSequenceClassification(TinyBertPretrainedModel):
    """
    TinyBert Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        tinybert (:class:`TinyBertModel`):
            An instance of TinyBertModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of TinyBert.
            If None, use the same value as `hidden_dropout_prob` of `TinyBertModel`
            instance `tinybert`. Defaults to `None`.
    """

    def __init__(self, tinybert, num_classes=2, dropout=None):
        super(TinyBertForSequenceClassification, self).__init__()
        self.tinybert = tinybert
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.tinybert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.tinybert.config["hidden_size"],
                                    num_classes)
        self.activation = nn.ReLU()
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        r"""
        The TinyBertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`TinyBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`TinyBertModel`.
            attention_mask_list (list, optional):
                See :class:`TinyBertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.tinybert.modeling import TinyBertForSequenceClassification
                from paddlenlp.transformers.tinybert.tokenizer import TinyBertTokenizer

                tokenizer = TinyBertTokenizer.from_pretrained('tinybert-4l-312d')
                model = TinyBertForSequenceClassification.from_pretrained('tinybert-4l-312d')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP! ")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """

        sequence_output, pooled_output = self.tinybert(
            input_ids, token_type_ids, attention_mask)

        logits = self.classifier(self.activation(pooled_output))
        return logits
