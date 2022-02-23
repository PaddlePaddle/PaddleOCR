#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Copyright (c) 2021 ShannonAI

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import PretrainedModel, register_base_model
from paddlenlp.transformers.bert.modeling import BertPooler, BertPretrainingHeads

__all__ = [
    "ChineseBertModel",
    "ChineseBertPretrainedModel",
    "ChineseBertForPretraining",
    "ChineseBertPretrainingCriterion",
    "ChineseBertForSequenceClassification",
    "ChineseBertForTokenClassification",
    "ChineseBertForQuestionAnswering",
]


class PinyinEmbedding(nn.Layer):
    def __init__(self,
                 pinyin_map_len: int,
                 embedding_size: int,
                 pinyin_out_dim: int):
        """
        Pinyin Embedding Layer.

        Args:
            pinyin_map_len (int): the size of pinyin map, which about 26 Romanian characters and 6 numbers. 
            embedding_size (int): the size of each embedding vector.
            pinyin_out_dim (int): kernel number of conv.

        """
        super(PinyinEmbedding, self).__init__()

        self.pinyin_out_dim = pinyin_out_dim
        self.embedding = nn.Embedding(pinyin_map_len, embedding_size)
        self.conv = nn.Conv1D(
            in_channels=embedding_size,
            out_channels=self.pinyin_out_dim,
            kernel_size=2,
            stride=1,
            padding=0, )

    def forward(self, pinyin_ids):
        """
        Args:
            pinyin_ids (Tensor): Its shape is (bs*sentence_length*pinyin_locs).

        Returns:
            pinyin_embed (Tensor): Its shape is (bs,sentence_length,pinyin_out_dim).

        """
        # input pinyin ids for 1-D conv
        embed = self.embedding(
            pinyin_ids)  # [bs,sentence_length*pinyin_locs,embed_size]
        bs, sentence_length, pinyin_locs, embed_size = embed.shape
        view_embed = embed.reshape(
            shape=[-1, pinyin_locs,
                   embed_size])  # [(bs*sentence_length),pinyin_locs,embed_size]
        input_embed = view_embed.transpose(
            [0, 2, 1])  # [(bs*sentence_length), embed_size, pinyin_locs]
        # conv + max_pooling
        pinyin_conv = self.conv(
            input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        pinyin_embed = F.max_pool1d(
            pinyin_conv,
            pinyin_conv.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
        return pinyin_embed.reshape(
            shape=[bs, sentence_length,
                   self.pinyin_out_dim])  # [bs,sentence_length,pinyin_out_dim]


class GlyphEmbedding(nn.Layer):
    """Glyph2Image Embedding."""

    def __init__(self, num_embeddings, embedding_dim):
        super(GlyphEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, input_ids):
        """
        Get glyph images for batch inputs.

        Args:
            input_ids (Tensor): Its shape is [batch, sentence_length].

        Returns:
            images (Tensor): Its shape is [batch, sentence_length, self.font_num*self.font_size*self.font_size].
        
        """
        # return self.embedding(input_ids).reshape([-1, self.font_num, self.font_size, self.font_size])
        return self.embedding(input_ids)


class FusionBertEmbeddings(nn.Layer):
    """
    Construct the embeddings from word, position, glyph, pinyin and token_type embeddings.
    """

    def __init__(
            self,
            vocab_size,
            hidden_size,
            pad_token_id,
            type_vocab_size,
            max_position_embeddings,
            pinyin_map_len,
            glyph_embedding_dim,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1, ):
        super(FusionBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.pinyin_embeddings = PinyinEmbedding(
            pinyin_map_len=pinyin_map_len,
            embedding_size=128,
            pinyin_out_dim=hidden_size, )
        self.glyph_embeddings = GlyphEmbedding(vocab_size, glyph_embedding_dim)

        self.glyph_map = nn.Linear(glyph_embedding_dim, hidden_size)
        self.map_fc = nn.Linear(hidden_size * 3, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            paddle.expand(
                paddle.arange(
                    max_position_embeddings, dtype="int64"),
                shape=[1, -1]), )

    def forward(self,
                input_ids,
                pinyin_ids,
                token_type_ids=None,
                position_ids=None):

        input_shape = input_ids.shape
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")
        # get char embedding, pinyin embedding and glyph embedding
        word_embeddings = self.word_embeddings(input_ids)  # [bs,l,hidden_size]

        pinyin_embeddings = self.pinyin_embeddings(
            pinyin_ids.reshape(
                shape=[input_shape[0], seq_length, 8]))  # [bs,l,hidden_size]

        glyph_embeddings = self.glyph_map(
            self.glyph_embeddings(input_ids))  # [bs,l,hidden_size]
        # fusion layer
        concat_embeddings = paddle.concat(
            (word_embeddings, pinyin_embeddings, glyph_embeddings), axis=2)
        inputs_embeds = self.map_fc(concat_embeddings)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ChineseBertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained ChineseBert models. It provides ChineseBert related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    base_model_prefix = "chinesebert"
    model_config_file = "model_config.json"

    pretrained_init_configuration = {
        "ChineseBERT-base": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 23236,
            "glyph_embedding_dim": 1728,
            "pinyin_map_len": 32,
        },
        "ChineseBERT-large": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 23236,
            "glyph_embedding_dim": 1728,
            "pinyin_map_len": 32,
        },
    }

    resource_files_names = {"model_state": "model_state.pdparams"}

    pretrained_resource_files_map = {
        "model_state": {
            "ChineseBERT-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/chinese_bert/chinesebert-base/model_state.pdparams",
            "ChineseBERT-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/chinese_bert/chinesebert-large/model_state.pdparams",
        }
    }

    def init_weights(self, layer):
        """Initialize the weights."""

        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.chinesebert.config["initializer_range"],
                        shape=layer.weight.shape, ))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = (self.layer_norm_eps
                              if hasattr(self, "layer_norm_eps") else
                              self.chinesebert.config["layer_norm_eps"])


@register_base_model
class ChineseBertModel(ChineseBertPretrainedModel):
    """
    The bare ChineseBert Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `BChineseBertModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `ChineseBertModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layer and pooler layer. Defaults to `768`.
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
                See :meth:`BertPretrainedModel.init_weights()` for how weights are initialized in `BertModel`.

        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.

        pooled_act (str, optional):
            The non-linear activation function in the pooling layer.
            Defaults to `"tanh"`.
        
        layer_norm_eps
            The epsilon of layernorm.
            Defaults to `1e-12`.

        glyph_embedding_dim (int, optional):
            The dim of glyph_embedding.
            Defaults to `1728`.
        
        pinyin_map_len=32 (int, optional):
            The length of pinyin map.
            Defaults to `32`.

    """

    def __init__(
            self,
            vocab_size=23236,
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
            pool_act="tanh",
            layer_norm_eps=1e-12,
            glyph_embedding_dim=1728,
            pinyin_map_len=32, ):
        super(ChineseBertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.embeddings = FusionBertEmbeddings(
            vocab_size,
            hidden_size,
            pad_token_id,
            type_vocab_size,
            max_position_embeddings,
            pinyin_map_len,
            glyph_embedding_dim,
            layer_norm_eps,
            hidden_dropout_prob, )
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0, )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = BertPooler(hidden_size, pool_act)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                pinyin_ids=None,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                output_hidden_states=False):
        r'''
        The ChineseBert forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            pinyin_ids (Tensor, optional):
                Indices of input sequence tokens pinyin. We apply a CNN model with width 2 on the pinyin
                sequence, followed by max-pooling to derive the resulting pinyin embedding. This makes output
                dimensionality immune to the length of the input pinyin sequence. The length of the input pinyin 
                sequence is fixed at 8.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length, 8].
                Defaults to `None`, which means we don't add pinyin embeddings.
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
            output_hidden_states (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `False`.

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
                from paddlenlp.transformers import ChineseBertModel, ChineseBertTokenizer

                tokenizer = ChineseBertTokenizer.from_pretrained('ChineseBERT-base')
                model = ChineseBertModel.from_pretrained('ChineseBERT-base')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        '''

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e4,
                axis=[1, 2], )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            pinyin_ids=pinyin_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids, )

        if output_hidden_states:
            output = embedding_output
            encoder_outputs = []
            for mod in self.encoder.layers:
                output = mod(output, src_mask=attention_mask)
                encoder_outputs.append(output)
            if self.encoder.norm is not None:
                encoder_outputs[-1] = self.encoder.norm(encoder_outputs[-1])
            pooled_output = self.pooler(encoder_outputs[-1])
        else:
            sequence_output = self.encoder(embedding_output, attention_mask)
            pooled_output = self.pooler(sequence_output)
        if output_hidden_states:
            return encoder_outputs, pooled_output
        else:
            return sequence_output, pooled_output


class ChineseBertForQuestionAnswering(ChineseBertPretrainedModel):
    """
    ChineseBert Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        ChineseBert (:class:`ChineseBertModel`):
            An instance of ChineseBertModel.
        dropout (float, optional):
            The dropout probability for output of ChineseBert.
            If None, use the same value as `hidden_dropout_prob` of `ChineseBertModel`
            instance `chinesebert`. Defaults to `None`.
        """

    def __init__(self, chinesebert):
        super(ChineseBertForQuestionAnswering, self).__init__()
        self.chinesebert = chinesebert  # allow chinesebert to be config
        self.classifier = nn.Linear(self.chinesebert.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, pinyin_ids=None, token_type_ids=None):
        r"""
        The ChineseBertForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ChineseBertModel`.
            pinyin_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`ChineseBertModel`.

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
                from paddlenlp.transformers.chinesebert.modeling import ChineseBertForQuestionAnswering
                from paddlenlp.transformers.chinesebert.tokenizer import ChineseBertTokenizer

                tokenizer = ChineseBertTokenizer.from_pretrained('ChineseBERT-base')
                model = ChineseBertForQuestionAnswering.from_pretrained('ChineseBERT-base')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                start_logits = outputs[0]
                end_logits = outputs[1]
        """
        sequence_output, _ = self.chinesebert(
            input_ids,
            pinyin_ids,
            token_type_ids=token_type_ids,
            position_ids=None)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class ChineseBertForSequenceClassification(ChineseBertPretrainedModel):
    """
    ChineseBert Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        chinesebert (:class:`ChineseBertModel`):
            An instance of ChineseBertModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of ChineseBert.
            If None, use the same value as `hidden_dropout_prob` of `ChineseBertModel`
            instance `chinesebert`. Defaults to None.
    """

    def __init__(self, chinesebert, num_classes=2, dropout=None):
        super(ChineseBertForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.chinesebert = chinesebert  # allow chinesebert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  chinesebert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.chinesebert.config["hidden_size"],
                                    self.num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                pinyin_ids=None,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        The ChineseBertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ChineseBertModel`.
            pinyin_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            position_ids(Tensor, optional):
                See :class:`ChineseBertModel`.
            attention_mask (list, optional):
                See :class:`ChineseBertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.chinesebert.modeling import ChineseBertForSequenceClassification
                from paddlenlp.transformers.chinesebert.tokenizer import ChineseBertTokenizer

                tokenizer = ChineseBertTokenizer.from_pretrained('ChineseBERT-base')
                model = ChineseBertForSequenceClassification.from_pretrained('ChineseBERT-base', num_classes=2)

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
                print(logits.shape)
                # [1, 2]

        """

        _, pooled_output = self.chinesebert(
            input_ids,
            pinyin_ids=pinyin_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask, )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class ChineseBertForTokenClassification(ChineseBertPretrainedModel):
    """
    ChineseBert Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        chinesebert (:class:`ChineseBertModel`):
            An instance of ChineseBertModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of ChineseBert.
            If None, use the same value as `hidden_dropout_prob` of `ChineseBertModel`
            instance `chinesebert`. Defaults to None.
    """

    def __init__(self, chinesebert, num_classes=2, dropout=None):
        super(ChineseBertForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.chinesebert = chinesebert  # allow chinesebert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  chinesebert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.chinesebert.config["hidden_size"],
                                    self.num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                pinyin_ids=None,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        The ChineseBertForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ChineseBertModel`.
            pinyin_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            position_ids(Tensor, optional):
                See :class:`ChineseBertModel`.
            attention_mask (list, optional):
                See :class:`ChineseBertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
             .. code-block::

                import paddle
                from paddlenlp.transformers.chinesebert.modeling import ChineseBertForSequenceClassification
                from paddlenlp.transformers.chinesebert.tokenizer import ChineseBertTokenizer

                tokenizer = ChineseBertTokenizer.from_pretrained('ChineseBERT-base')
                model = ChineseBertForSequenceClassification.from_pretrained('ChineseBERT-base', num_classes=2)

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
                print(logits.shape)
                # [1, 13, 2]

        """
        sequence_output, _ = self.chinesebert(
            input_ids,
            pinyin_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask, )

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class ChineseBertForPretraining(ChineseBertPretrainedModel):
    """
    ChineseBert Model with pretraining tasks on top.

    Args:
        chinesebert (:class:`ChineseBertModel`):
            An instance of :class:`ChineseBertModel`.

    """

    def __init__(self, chinesebert):
        super(ChineseBertForPretraining, self).__init__()
        self.chinesebert = chinesebert
        self.cls = BertPretrainingHeads(
            self.chinesebert.config["hidden_size"],
            self.chinesebert.config["vocab_size"],
            self.chinesebert.config["hidden_act"],
            embedding_weights=self.chinesebert.embeddings.word_embeddings.
            weight, )

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                pinyin_ids=None,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`ChineseBertModel`.
            pinyin_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            position_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            attention_mask (Tensor, optional):
                See :class:`ChineseBertModel`.
            masked_positions(Tensor, optional):
                See :class:`ChineseBertPretrainingHeads`.

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

        """
        with paddle.static.amp.fp16_guard():
            outputs = self.chinesebert(
                input_ids,
                pinyin_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask, )
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_positions)
            return prediction_scores, seq_relationship_score


class ChineseBertPretrainingCriterion(nn.Layer):
    """

    Args:
        vocab_size(int):
            Vocabulary size of `inputs_ids` in `ChineseBertModel`. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling `ChineseBertBertModel`.

    """

    def __init__(self, vocab_size):
        super(ChineseBertPretrainingCriterion, self).__init__()
        # CrossEntropyLoss is expensive since the inner reshape (copy)
        self.loss_fn = nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score,
                masked_lm_labels, next_sentence_labels, masked_lm_scale):
        """
        Args:
            prediction_scores(Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size]
            seq_relationship_score(Tensor):
                The scores of next sentence prediction. Its data type should be float32 and
                its shape is [batch_size, 2]
            masked_lm_labels(Tensor):
                The labels of the masked language modeling, its dimensionality is equal to `prediction_scores`.
                Its data type should be int64. If `masked_positions` is None, its shape is [batch_size, sequence_length, 1].
                Otherwise, its shape is [batch_size, mask_token_num, 1]
            next_sentence_labels(Tensor):
                The labels of the next sentence prediction task, the dimensionality of `next_sentence_labels`
                is equal to `seq_relation_labels`. Its data type should be int64 and
                its shape is [batch_size, 1]
            masked_lm_scale(Tensor or int):
                The scale of masked tokens. Used for the normalization of masked language modeling loss.
                If it is a `Tensor`, its data type should be int64 and its shape is equal to `prediction_scores`.

        Returns:
            Tensor: The pretraining loss, equals to the sum of `masked_lm_loss` plus the mean of `next_sentence_loss`.
            Its data type should be float32 and its shape is [1].


        """
        with paddle.static.amp.fp16_guard():
            masked_lm_loss = F.cross_entropy(
                prediction_scores,
                masked_lm_labels,
                reduction="none",
                ignore_index=-1)
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            next_sentence_loss = F.cross_entropy(
                seq_relationship_score, next_sentence_labels, reduction="none")
        return paddle.sum(masked_lm_loss) + paddle.mean(next_sentence_loss)
