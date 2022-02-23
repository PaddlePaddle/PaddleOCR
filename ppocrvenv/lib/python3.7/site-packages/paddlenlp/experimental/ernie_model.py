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

from paddlenlp.experimental import FasterTokenizer, FasterPretrainedModel
from paddlenlp.transformers.model_utils import register_base_model
from paddlenlp.transformers.ernie.modeling import ErnieEmbeddings, ErniePooler

__all__ = [
    "FasterErnieModel", "FasterErnieForSequenceClassification",
    "FasterErnieForTokenClassification"
]


class FasterErniePretrainedModel(FasterPretrainedModel):
    r"""
    An abstract class for pretrained ERNIE models. It provides ERNIE related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. 
    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "ernie-1.0": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 513,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 18000,
            "pad_token_id": 0,
            "do_lower_case": True,
        },
        "ernie-2.0-en": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
            "do_lower_case": True,
        },
        "ernie-2.0-en-finetuned-squad": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
            "do_lower_case": True,
        },
        "ernie-2.0-large-en": {
            "attention_probs_dropout_prob": 0.1,
            "intermediate_size": 4096,  # special for ernie-2.0-large-en
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
            "do_lower_case": True,
        },
    }
    resource_files_names = {
        "model_state": "model_state.pdparams",
        "vocab_file": "vocab.txt"
    }
    pretrained_resource_files_map = {
        "model_state": {
            "ernie-1.0":
            "https://bj.bcebos.com/paddlenlp/models/transformers/faster_ernie/faster_ernie_v1_chn_base.pdparams",
            "ernie-2.0-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/faster_ernie_v2_base/faster_ernie_v2_eng_base.pdparams",
            "ernie-2.0-en-finetuned-squad":
            "https://bj.bcebos.com/paddlenlp/models/transformers/faster_ernie_v2_base/faster_ernie_v2_eng_base_finetuned_squad.pdparams",
            "ernie-2.0-large-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/faster_ernie_v2_large/faster_ernie_v2_eng_large.pdparams",
        },
        "vocab_file": {
            "ernie-1.0":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie/vocab.txt",
            "ernie-2.0-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_v2_base/vocab.txt",
            "ernie-2.0-en-finetuned-squad":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_v2_base/vocab.txt",
            "ernie-2.0-large-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_v2_large/vocab.txt",
        }
    }
    base_model_prefix = "ernie"

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
                        self.ernie.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class FasterErnieModel(FasterErniePretrainedModel):
    r"""
    The bare ERNIE Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `ErnieModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `ErnieModel`.
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
                See :meth:`ErniePretrainedModel._init_weights()` for how weights are initialized in `ErnieModel`.

        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.

    """

    def __init__(
            self,
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
            max_seq_len=512, ):
        super(FasterErnieModel, self).__init__()
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`model = FasterErnieModel.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.do_lower_case = do_lower_case
        self.vocab = self.load_vocabulary(vocab_file)
        self.max_seq_len = max_seq_len

        self.tokenizer = FasterTokenizer(
            self.vocab,
            do_lower_case=self.do_lower_case,
            is_split_into_words=is_split_into_words)
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
            mean=0.0, std=self.initializer_range))
        self.embeddings = ErnieEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size, pad_token_id, weight_attr)
        # Avoid import error in global scope when using paddle <= 2.2.0, therefore
        # import FusedTransformerEncoderLayer in local scope.
        # FusedTransformerEncoderLayer is supported by paddlepaddle since 2.2.0, please
        # ensure the version >= 2.2.0
        from paddle.incubate.nn import FusedTransformerEncoderLayer
        encoder_layer = FusedTransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout_rate=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout_rate=attention_probs_dropout_prob,
            act_dropout_rate=0,
            weight_attr=weight_attr, )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = ErniePooler(hidden_size, weight_attr)
        self.apply(self.init_weights)

    def forward(self, text, text_pair=None):
        input_ids, token_type_ids = self.tokenizer(
            text=text, text_pair=text_pair, max_seq_len=self.max_seq_len)

        attention_mask = paddle.unsqueeze(
            (input_ids == self.pad_token_id
             ).astype(self.pooler.dense.weight.dtype) * -1e4,
            axis=[1, 2])
        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class FasterErnieForSequenceClassification(FasterErniePretrainedModel):
    def __init__(self, ernie, num_classes=2, dropout=None):
        super(FasterErnieForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self, text, text_pair=None):

        _, pooled_output = self.ernie(text, text_pair)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        predictions = paddle.argmax(logits, axis=-1)
        return logits, predictions


class FasterErnieForTokenClassification(FasterErniePretrainedModel):
    def __init__(self, ernie, num_classes=2, dropout=None):
        super(FasterErnieForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self, text, text_pair=None):

        sequence_output, _ = self.ernie(text, text_pair)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        predictions = paddle.argmax(logits, axis=-1)
        return logits, predictions
