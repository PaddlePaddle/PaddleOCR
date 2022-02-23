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
import os

import paddle
import paddle.fluid.core as core
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.utils.log import logger
from paddlenlp.experimental import FasterTokenizer, FasterPretrainedModel
from .. import PretrainedModel, register_base_model

__all__ = [
    'PPMiniLMModel',
    'PPMiniLMPretrainedModel',
    'PPMiniLMForSequenceClassification',
]


class PPMiniLMEmbeddings(nn.Layer):
    r"""
    Include embeddings from word, position and token_type embeddings.
    """

    def __init__(
            self,
            vocab_size,
            hidden_size=768,
            hidden_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            pad_token_id=0,
            weight_attr=None, ):
        super(PPMiniLMEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=pad_token_id,
            weight_attr=weight_attr)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size, weight_attr=weight_attr)
        self.token_type_embeddings = nn.Embedding(
            type_vocab_size, hidden_size, weight_attr=weight_attr)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            # maybe need use shape op to unify static graph and dynamic graph
            #seq_length = input_ids.shape[1]
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")
        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PPMiniLMPooler(nn.Layer):
    def __init__(self, hidden_size, weight_attr=None):
        super(PPMiniLMPooler, self).__init__()
        self.dense = nn.Linear(
            hidden_size, hidden_size, weight_attr=weight_attr)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class PPMiniLMPretrainedModel(FasterPretrainedModel):
    r"""
    An abstract class for pretrained PPMiniLM models. It provides PPMiniLM related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models. 
    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "ppminilm-6l-768h": {
            "attention_probs_dropout_prob": 0.1,
            "intermediate_size": 3072,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "type_vocab_size": 4,
            "vocab_size": 21128,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {
        "model_state": "model_state.pdparams",
        "vocab_file": "vocab.txt"
    }
    pretrained_resource_files_map = {
        "model_state": {
            "ppminilm-6l-768h":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ppminilm-6l-768h/ppminilm-6l-768h.pdparams",
        },
        "vocab_file": {
            "ppminilm-6l-768h":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ppminilm-6l-768h/vocab.txt",
        }
    }
    base_model_prefix = "ppminilm"
    use_faster_tokenizer = False

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
                        self.ppminilm.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12

    def add_faster_tokenizer_op(self):
        self.ppminilm.tokenizer = FasterTokenizer(
            self.ppminilm.vocab,
            do_lower_case=self.ppminilm.do_lower_case,
            is_split_into_words=self.ppminilm.is_split_into_words)

    def to_static(self,
                  output_path,
                  use_faster_tokenizer=True,
                  is_text_pair=False):
        self.eval()
        self.use_faster_tokenizer = use_faster_tokenizer
        # Convert to static graph with specific input description
        if self.use_faster_tokenizer:
            self.add_faster_tokenizer_op()
            if is_text_pair:
                model = paddle.jit.to_static(
                    self,
                    input_spec=[
                        paddle.static.InputSpec(
                            shape=[None],
                            dtype=core.VarDesc.VarType.STRINGS,
                            name="text"), paddle.static.InputSpec(
                                shape=[None],
                                dtype=core.VarDesc.VarType.STRINGS,
                                name="text_pair")
                    ])
            else:
                model = paddle.jit.to_static(
                    self,
                    input_spec=[
                        paddle.static.InputSpec(
                            shape=[None],
                            dtype=core.VarDesc.VarType.STRINGS,
                            name="text")
                    ])
        else:
            model = paddle.jit.to_static(
                self,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None, None], dtype="int64",
                        name="input_ids"),  # input_ids
                    paddle.static.InputSpec(
                        shape=[None, None],
                        dtype="int64",
                        name="token_type_ids")  # segment_ids
                ])
        paddle.jit.save(model, output_path)
        logger.info("Already save the static model to the path %s" %
                    output_path)


@register_base_model
class PPMiniLMModel(PPMiniLMPretrainedModel):
    r"""
    The bare PPMiniLM Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `PPMiniLMModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `PPMiniLMModel`.
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
                See :meth:`PPMiniLMPretrainedModel._init_weights()` for how weights are initialized in `PPMiniLMModel`.

        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.

    """

    def __init__(self,
                 vocab_size,
                 vocab_file,
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
                 do_lower_case=True,
                 is_split_into_words=False,
                 max_seq_len=128,
                 pad_to_max_seq_len=False):
        super(PPMiniLMModel, self).__init__()
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`model = PPMiniLMModel.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.vocab = self.load_vocabulary(vocab_file)
        self.do_lower_case = do_lower_case
        self.max_seq_len = max_seq_len
        self.is_split_into_words = is_split_into_words
        self.pad_token_id = pad_token_id
        self.pad_to_max_seq_len = pad_to_max_seq_len
        self.initializer_range = initializer_range
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.TruncatedNormal(
                mean=0.0, std=self.initializer_range))
        self.embeddings = PPMiniLMEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size, pad_token_id, weight_attr)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
            weight_attr=weight_attr,
            normalize_before=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = PPMiniLMPooler(hidden_size, weight_attr)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        Args:
            input_ids (Tensor, List[string]):
                If `input_ids` is a Tensor object, it is an indices of input
                sequence tokens in the vocabulary. They are numerical
                representations of tokens that build the input sequence. It's
                data type should be `int64` and has a shape of [batch_size, sequence_length].
                If `input_ids` is a list of string, `self.use_faster_tokenizer`
                should be True, and the network contains `faster_tokenizer`
                operator.
            token_type_ids (Tensor, string, optional):
                If `token_type_ids` is a Tensor object:
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.

                If `token_type_ids` is a list of string: `self.use_faster_tokenizer`
                should be True, and the network contains `faster_tokenizer` operator.

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
                We use whole-word-mask in PPMiniLM, so the whole word will have the same value. For example, "使用" as a word,
                "使" and "用" will have the same value.
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
                from paddlenlp.transformers import PPMiniLMModel, PPMiniLMTokenizer

                tokenizer = PPMiniLMTokenizer.from_pretrained('ppminilm-6l-768h')
                model = PPMiniLMModel.from_pretrained('ppminilm-6l-768h')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                sequence_output, pooled_output = model(**inputs)

        """
        # Only for saving
        if self.use_faster_tokenizer:
            input_ids, token_type_ids = self.tokenizer(
                text=input_ids,
                text_pair=token_type_ids,
                max_seq_len=self.max_seq_len,
                pad_to_max_seq_len=self.pad_to_max_seq_len)
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
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


class PPMiniLMForSequenceClassification(PPMiniLMPretrainedModel):
    r"""
    PPMiniLM Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        ppminilm (PPMiniLMModel): 
            An instance of `paddlenlp.transformers.PPMiniLMModel`.
        num_classes (int, optional): 
            The number of classes. Default to `2`.
        dropout (float, optional): 
            The dropout probability for output of PPMiniLM. 
            If None, use the same value as `hidden_dropout_prob` 
            of `paddlenlp.transformers.PPMiniLMModel` instance. Defaults to `None`.
    """

    def __init__(self, ppminilm, num_classes=2, dropout=None):
        super(PPMiniLMForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.ppminilm = ppminilm  # allow ppminilm to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ppminilm.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ppminilm.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`PPMiniLMModel`.
            token_type_ids (Tensor, optional):
                See :class:`PPMiniLMModel`.
            position_ids (Tensor, optional):
                See :class:`PPMiniLMModel`.
            attention_mask (Tensor, optional):
                See :class:`MiniLMModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import PPMiniLMForSequenceClassification, PPMiniLMTokenizer

                tokenizer = PPMiniLMTokenizer.from_pretrained('ppminilm-6l-768h')
                model = PPMiniLMForSequenceClassification.from_pretrained('ppminilm-6l-768h0')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        self.ppminilm.use_faster_tokenizer = self.use_faster_tokenizer
        _, pooled_output = self.ppminilm(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
