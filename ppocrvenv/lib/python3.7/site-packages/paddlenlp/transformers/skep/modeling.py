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

from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss
from paddlenlp.utils.log import logger
from paddlenlp.utils.tools import compare_version

if compare_version(paddle.version.full_version, "2.2.0") >= 0:
    # paddle.text.ViterbiDecoder is supported by paddle after version 2.2.0
    from paddle.text import ViterbiDecoder
else:
    from paddlenlp.layers.crf import ViterbiDecoder
from .. import PretrainedModel, register_base_model

__all__ = [
    'SkepModel', 'SkepPretrainedModel', 'SkepForSequenceClassification',
    'SkepForTokenClassification', 'SkepCrfForTokenClassification'
]


class SkepEmbeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 pad_token_id=0):
        super(SkepEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.type_vocab_size = type_vocab_size
        if self.type_vocab_size != 0:
            self.token_type_embeddings = nn.Embedding(type_vocab_size,
                                                      hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            # maybe need use shape op to unify static graph and dynamic graph
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embedings + position_embeddings
        if self.type_vocab_size != 0:
            if token_type_ids is None:
                token_type_ids = paddle.zeros_like(input_ids, dtype="int64")
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings
        elif token_type_ids is not None:
            logger.warning(
                "There is no need to pass the token type ids to SKEP based on RoBERTa model."
                "The input token type ids will be ignored.")

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SkepPooler(nn.Layer):
    """
    The pooling layer on skep model.
    """

    def __init__(self, hidden_size):
        super(SkepPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SkepPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained Skep models. It provides Skep related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "skep_ernie_1.0_large_ch": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,  # special for ernie-large
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 4,
            "vocab_size": 12800,
            "pad_token_id": 0,
        },
        "skep_ernie_2.0_large_en": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,  # special for ernie-large
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
        },
        "skep_roberta_large_en": {
            "attention_probs_dropout_prob": 0.1,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "max_position_embeddings": 514,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 0,
            "vocab_size": 50265,
            "pad_token_id": 1,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "skep_ernie_1.0_large_ch":
            "https://bj.bcebos.com/paddlenlp/models/transformers/skep/skep_ernie_1.0_large_ch.pdparams",
            "skep_ernie_2.0_large_en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/skep/skep_ernie_2.0_large_en.pdparams",
            "skep_roberta_large_en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/skep/skep_roberta_large_en.pdparams",
        }
    }
    base_model_prefix = "skep"

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
                        self.skep.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-5


@register_base_model
class SkepModel(SkepPretrainedModel):
    r"""
    The bare SKEP Model outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    More details refer to `SKEP <https://www.aclweb.org/anthology/2020.acl-main.374>`.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `SKEPModel`. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling `SKEPModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layers and the pooler layer. Defaults to `768`.
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
            are supported. Defaults to ``"gelu"``.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to ``0.1``.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding. The dimensionality of position encoding
            is the dimensionality of the sequence in `TinyBertModel`.
            Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids` passed when calling `~transformers.SkepModel`.
            Defaults to `2`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer.
            Defaults to `0.02`.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`SkepPretrainedModel.init_weights()` for how weights are initialized in `SkepModel`.

        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.

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
                 type_vocab_size=2,
                 initializer_range=0.02,
                 pad_token_id=0):
        super(SkepModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = SkepEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size, pad_token_id)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = SkepPooler(hidden_size)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        The SkepModel forward method, overrides the `__call__()` special method.

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
            position_ids (Tensor, optionals):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            attention_mask (`Tensor`, optional):
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
            tuple: Returns tuple (`sequence_output`, `pooled_output`).

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
                from paddlenlp.transformers import SkepModel, SkepTokenizer

                tokenizer = SkepTokenizer.from_pretrained('skep_ernie_2.0_large_en')
                model = SkepModel.from_pretrained('skep_ernie_2.0_large_en')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP! ")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)

        """
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids.astype("int64") == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e4,
                axis=[1, 2])
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class SkepForSequenceClassification(SkepPretrainedModel):
    r"""
    SKEP Model with a linear layer on top of the pooled output,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        skep (:class:`SkepModel`):
            An instance of SkepModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of SKEP.
            If None, use the same value as `hidden_dropout_prob` of `SkepModel`
            instance `skep`. Defaults to None.

    """

    def __init__(self, skep, num_classes=2, dropout=None):
        super(SkepForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.skep = skep  # allow skep to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.skep.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.skep.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        The SkepForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`SkepModel`.
            token_type_ids (Tensor, optional):
                See :class:`SkepModel`.
            position_ids (Tensor, `optional`):
                See :class:`SkepModel`.
            attention_mask (Tensor, optional):
                See :class:`SkepModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

                tokenizer = SkepTokenizer.from_pretrained('skep_ernie_2.0_large_en')
                model = SkepForSequenceClassification.from_pretrained('skep_ernie_2.0_large_en')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        _, pooled_output = self.skep(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class SkepForTokenClassification(SkepPretrainedModel):
    r"""
    SKEP Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        skep (:class:`SkepModel`):
            An instance of SkepModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of SKEP.
            If None, use the same value as `hidden_dropout_prob` of `SkepModel`
            instance `skep`. Defaults to None.

    """

    def __init__(self, skep, num_classes=2, dropout=None):
        super(SkepForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.skep = skep  # allow skep to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.skep.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.skep.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        The SkepForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`SkepModel`.
            token_type_ids (Tensor, optional):
                See :class:`SkepModel`.
            position_ids (Tensor, optional):
                See :class:`SkepModel`.
            attention_mask (Tensor, optional):
                See :class:`SkepModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import SkepForTokenClassification, SkepTokenizer

                tokenizer = SkepTokenizer.from_pretrained('skep_ernie_2.0_large_en')
                model = SkepForTokenClassification.from_pretrained('skep_ernie_2.0_large_en')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        sequence_output, _ = self.skep(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class SkepCrfForTokenClassification(nn.Layer):
    r"""
    SKEPCRF Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        skep (:class:`SkepModel`):
            An instance of SkepModel.
        num_classes (int):
            The number of classes.

    """

    def __init__(self, skep, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.skep = skep  # allow skep to be config
        gru_hidden_size = 128

        self.gru = nn.GRU(self.skep.config["hidden_size"],
                          gru_hidden_size,
                          num_layers=2,
                          direction='bidirect')
        self.fc = nn.Linear(
            gru_hidden_size * 2,
            self.num_classes,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(
                    low=-0.1, high=0.1),
                regularizer=paddle.regularizer.L2Decay(coeff=1e-4)))
        self.crf = LinearChainCrf(
            self.num_classes, crf_lr=0.2, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions, False)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                seq_lens=None,
                labels=None):
        r"""
        The SkepCrfForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`SkepModel`.
            token_type_ids (Tensor, optional):
                See :class:`SkepModel`.
            position_ids (Tensor, optional):
                See :class:`SkepModel`.
            attention_mask (Tensor, optional):
                See :class:`SkepModel`.
            seq_lens (Tensor, optional):
                The input length tensor storing real length of each sequence for correctness.
                Its data type should be int64 and its shape is `[batch_size]`.
                Defaults to `None`.
            labels (Tensor, optional):
                The input label tensor.
                Its data type should be int64 and its shape is `[batch_size, sequence_length]`.

        Returns:
            Tensor: Returns tensor `loss` if `labels` is not None. Otherwise, returns tensor `prediction`.

            - `loss` (Tensor):
                The crf loss. Its data type is float32 and its shape is `[batch_size]`.

            - `prediction` (Tensor):
                The prediction tensor containing the highest scoring tag indices.
                Its data type is int64 and its shape is `[batch_size, sequence_length]`.

        """
        sequence_output, _ = self.skep(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        bigru_output, _ = self.gru(
            sequence_output)  #, sequence_length=seq_lens)
        emission = self.fc(bigru_output)

        if labels is not None:
            loss = self.crf_loss(emission, seq_lens, labels)
            return loss
        else:
            _, prediction = self.viterbi_decoder(emission, seq_lens)
            return prediction
