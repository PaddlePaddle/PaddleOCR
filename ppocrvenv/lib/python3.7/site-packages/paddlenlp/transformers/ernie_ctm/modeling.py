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
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.nn import TransformerEncoder, Linear, Layer, Embedding, LayerNorm, Tanh
from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss
from paddlenlp.utils.tools import compare_version

if compare_version(paddle.version.full_version, "2.2.0") >= 0:
    # paddle.text.ViterbiDecoder is supported by paddle after version 2.2.0
    from paddle.text import ViterbiDecoder
else:
    from paddlenlp.layers.crf import ViterbiDecoder

from .. import PretrainedModel, register_base_model

__all__ = [
    'ErnieCtmPretrainedModel', 'ErnieCtmModel', 'ErnieCtmWordtagModel',
    'ErnieCtmNptagModel', 'ErnieCtmForTokenClassification'
]


class ErnieCtmEmbeddings(Layer):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 padding_idx=0,
                 cls_num=2):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_size, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                embedding_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size,
                                                  embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.cls_num = cls_num

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)

            content_len = paddle.shape(input_ids)[1] - self.cls_num
            position_ids = paddle.concat([
                paddle.zeros(
                    shape=[self.cls_num], dtype="int64"), paddle.linspace(
                        1, content_len, content_len, dtype="int64")
            ])
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + token_type_embeddings + position_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class ErnieCtmPooler(Layer):
    """
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ErnieCtmPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained ErnieCtm models. It provides ErnieCtm related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading
     and loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "ernie-ctm": {
            "vocab_size": 23000,
            "embedding_size": 128,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "use_content_summary": True,
            "content_summary_index": 1,
            "cls_num": 2,
        },
        "wordtag": {
            "vocab_size": 23000,
            "embedding_size": 128,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "use_content_summary": True,
            "content_summary_index": 1,
            "cls_num": 2,
        },
        "nptag": {
            "vocab_size": 23000,
            "embedding_size": 128,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "use_content_summary": True,
            "content_summary_index": 1,
            "cls_num": 2,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "ernie-ctm":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_ctm/ernie_ctm_base_pos.pdparams",
            "wordtag":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_ctm/wordtag_pos.pdparams",
            "nptag":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_ctm/nptag.pdparams",
        }
    }
    base_model_prefix = "ernie_ctm"

    def init_weights(self, layer):
        # Initialize weights
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.ernie_ctm.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class ErnieCtmModel(ErnieCtmPretrainedModel):
    """
    The bare ErnieCtm Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `ErnieCtmModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `ErnieCtmModel`.
        embedding_size (int, optional):
            Dimensionality of the embedding layer.
            Defaults to `128`.
        hidden_size (int, optional):
            Dimensionality of the encoder layers and the pooler layer.
            Defaults to `768`.
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
            Defaults to `16`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer for initializing all weight matrices.
            Defaults to `0.02`.
        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        use_content_summary (`bool`, optional):
            Whether or not to add content summary tokens.
            Defaults to `True`.
        content_summary_index (int, optional):
            The number of the content summary tokens. Only valid when use_content_summary is True.
            Defaults to `1`.
        cls_num (int, optional):
            The number of the CLS tokens. Only valid when use_content_summary is True.
            Defaults to `2`.
    """

    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0,
                 use_content_summary=True,
                 content_summary_index=1,
                 cls_num=2):
        super(ErnieCtmModel, self).__init__()

        self.pad_token_id = pad_token_id
        self.content_summary_index = content_summary_index
        self.initializer_range = initializer_range
        self.embeddings = ErnieCtmEmbeddings(
            vocab_size,
            embedding_size,
            hidden_dropout_prob=hidden_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            padding_idx=pad_token_id,
            cls_num=cls_num)
        self.embedding_hidden_mapping_in = nn.Linear(embedding_size,
                                                     hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation="gelu",
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        encoder_layer.activation = nn.GELU(approximate=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = ErnieCtmPooler(hidden_size)

        self.use_content_summary = use_content_summary
        self.content_summary_index = content_summary_index
        if use_content_summary is True:
            self.feature_fuse = nn.Linear(hidden_size * 2, intermediate_size)
            self.feature_output = nn.Linear(intermediate_size, hidden_size)

        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                content_clone=False):
        r"""
        The ErnieCtmModel forward method, overrides the __call__() special method.
        
        Args:
            input_ids (`Tensor`):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            token_type_ids (`Tensor`, optional):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
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
                We use whole-word-mask in ERNIE, so the whole word will have the same value. For example, "使用" as a word,
                "使" and "用" will have the same value.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            content_clone (bool, optional):
                Whether the `content_output` is clone from `sequence_output`. If set to `True`, the content_output is
                clone from sequence_output, which may cause the classification task impact on the sequence labeling task.
                Defaults to `False`.

        Returns:
            tuple: Returns tuple (``sequence_output``, ``pooled_output``, ``content_output``).
            
            With the fields:

            - `sequence_output` (Tensor):
                Sequence of output at the last layer of the model. Its data type should be float32 and
                has a shape of [batch_size, sequence_length, hidden_size].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].

            - `content_output` (Tensor):
                The output of content summary token (`[CLS1]` in sequence). Its data type should be float32 and
                has a shape of [batch_size, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieModel, ErnieTokenizer

                tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
                model = ErnieModel.from_pretrained('ernie-1.0')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                sequence_output, pooled_output, content_output = model(**inputs)

        """
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e4,
                axis=[1, 2])
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(
                attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
        attention_mask.stop_gradient = True

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        embedding_output = self.embedding_hidden_mapping_in(embedding_output)

        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs

        pooled_output = self.pooler(sequence_output)
        content_output = (sequence_output[:, self.content_summary_index]
                          if self.use_content_summary else None)

        if self.use_content_summary is True:
            if content_clone is True:
                sequence_output = paddle.concat(
                    (sequence_output,
                     sequence_output[:, self.content_summary_index].clone(
                     ).unsqueeze([1]).expand_as(sequence_output)), 2)
            else:
                content_output = paddle.expand(
                    content_output.unsqueeze([1]),
                    shape=(sequence_output.shape[0], sequence_output.shape[1],
                           sequence_output.shape[2]))

                sequence_output = paddle.concat(
                    (sequence_output, content_output), 2)

            sequence_output = self.feature_fuse(sequence_output)

            sequence_output = self.feature_output(sequence_output)

        return sequence_output, pooled_output, content_output


class ErnieCtmWordtagModel(ErnieCtmPretrainedModel):
    """
    ErnieCtmWordtag Model with a token classification head on top (a crf layer on top of the hidden-states output) .
    e.g. for Named-Entity-Recognition (NER) tasks.

    Args:
        ernie_ctm (:clss:`ErnieCtmModel`):
            An instance of :class:`ErnieCtmModel`.
        num_tag (int):
            The number of different tags.
        crf_lr (float):
            The learning rate of the crf. Defaults to `100`.
    """

    def __init__(self, ernie_ctm, num_tag, crf_lr=100):
        super(ErnieCtmWordtagModel, self).__init__()
        self.num_tag = num_tag
        self.ernie_ctm = ernie_ctm
        self.tag_classifier = nn.Linear(self.ernie_ctm.config["hidden_size"],
                                        self.num_tag)
        self.crf = LinearChainCrf(
            self.num_tag, crf_lr, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions, False)

        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                lengths=None,
                position_ids=None,
                attention_mask=None,
                tag_labels=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieCtmModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieCtmModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieCtmModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieCtmModel`.
            lengths (Tensor, optional):
                The input length. Its dtype is int64 and has a shape of `[batch_size]`.
                Defaults to `None`.
            tag_labels (Tensor, optional):
                The input predicted tensor.
                Its dtype is float32 and has a shape of `[batch_size, sequence_length, num_tags]`.
                Defaults to `None`.

        Returns:
            tuple: Returns tuple (`seq_logits`, `cls_logits`).

            With the fields:

            - `seq_logits` (Tensor):
                A tensor of next sentence prediction logits.
                Its data type should be float32 and its shape is [batch_size, sequence_length, num_tag].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieCtmWordtagModel, ErnieCtmTokenizer

                tokenizer = ErnieCtmTokenizer.from_pretrained('ernie-ctm')
                model = ErnieCtmWordtagModel.from_pretrained('ernie-ctm', num_tag=2)

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        outputs = self.ernie_ctm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = outputs[0]
        seq_logits = self.tag_classifier(sequence_output)
        if tag_labels is not None:
            seq_crf_loss = self.crf_loss(seq_logits, lengths, tag_labels)
            return seq_crf_loss, seq_logits
        else:
            _, prediction = self.viterbi_decoder(seq_logits, lengths)
            return prediction


class ErnieCtmMLMHead(Layer):
    def __init__(self, embedding_size, vocab_size, hidden_size):
        super(ErnieCtmMLMHead, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_size)

        self.bias = self.create_parameter(
            [vocab_size],
            is_bias=True,
            default_initializer=nn.initializer.Constant(value=0.0))
        self.dense = nn.Linear(hidden_size, embedding_size)
        self.decoder = nn.Linear(embedding_size, vocab_size)
        self.activation = nn.GELU(approximate=True)
        # Link bias
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        prediction_scores = hidden_states
        return prediction_scores


class ErnieCtmNptagModel(ErnieCtmPretrainedModel):
    r"""
    ErnieCtmNptag Model with a `masked language modeling` head on top.

    Args:
        ernie_ctm (:clss:`ErnieCtmModel`):
            An instance of :class:`ErnieCtmModel`.
    """

    def __init__(self, ernie_ctm):
        super(ErnieCtmNptagModel, self).__init__()

        self.ernie_ctm = ernie_ctm
        self.predictions = ErnieCtmMLMHead(
            self.ernie_ctm.config["embedding_size"],
            self.ernie_ctm.config["vocab_size"],
            self.ernie_ctm.config["hidden_size"])

        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieCtmModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieCtmModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieCtmModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieCtmModel`.

        Returns:
            tuple: Returns tensor `logits`, the scores of masked token prediction.
            Its data type should be float32 and shape is [batch_size, sequence_length, vocab_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieCtmNptagModel, ErnieCtmTokenizer

                tokenizer = ErnieCtmTokenizer.from_pretrained('ernie-ctm')
                model = ErnieCtmNptagModel.from_pretrained('ernie-ctm')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                
                logits = model(**inputs)
                print(logits.shape)
                # [1, 45, 23000]

        """
        outputs = self.ernie_ctm(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids)
        sequence_output = outputs[0]
        logits = self.predictions(sequence_output)
        return logits


class ErnieCtmForTokenClassification(ErnieCtmPretrainedModel):
    r"""
    ERNIECtm Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        ernie (`ErnieModel`):
            An instance of `ErnieModel`.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of ERNIE.
            If None, use the same value as `hidden_dropout_prob`
            of `ErnieCtmModel` instance `ernie`. Defaults to `None`.
    """

    def __init__(self, ernie_ctm, num_classes=2, dropout=None):
        super(ErnieCtmForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie_ctm = ernie_ctm  # allow ernie_ctm to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie_ctm.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie_ctm.config["hidden_size"],
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
                See :class:`ErnieCtmModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieCtmModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieCtmModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieCtmModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[sequence_length, num_classes]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieCtmForTokenClassification, ErnieCtmTokenizer

                tokenizer = ErnieCtmTokenizer.from_pretrained('ernie-ctm')
                model = ErnieCtmForTokenClassification.from_pretrained('ernie-ctm')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """

        sequence_output, _, _ = self.ernie_ctm(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits
