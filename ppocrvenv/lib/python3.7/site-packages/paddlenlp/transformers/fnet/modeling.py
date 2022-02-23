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
"""Modeling classes for FNet model."""

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from functools import partial
from paddle.nn import Layer
from .. import PretrainedModel, register_base_model

__all__ = [
    "FNetPretrainedModel",
    "FNetModel",
    "FNetForSequenceClassification",
    "FNetForPreTraining",
    "FNetForMaskedLM",
    "FNetForNextSentencePrediction",
    "FNetForMultipleChoice",
    "FNetForTokenClassification",
    "FNetForQuestionAnswering"
]


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(
            activation_string, list(ACT2FN.keys())))


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
    return 0.5 * x * (1.0 + paddle.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))))


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


class FNetBasicOutput(Layer):
    def __init__(self, hidden_size, layer_norm_eps):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.layer_norm(input_tensor + hidden_states)
        return hidden_states


class FNetOutput(Layer):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 layer_norm_eps,
                 hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(input_tensor + hidden_states)
        return hidden_states


class FNetIntermediate(Layer):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class FNetLayer(Layer):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 layer_norm_eps,
                 hidden_dropout_prob,
                 hidden_act):
        super().__init__()
        self.fourier = FNetFourierTransform(hidden_size, layer_norm_eps)
        self.intermediate = FNetIntermediate(hidden_size, intermediate_size, hidden_act)
        self.output = FNetOutput(hidden_size,
                                 intermediate_size,
                                 layer_norm_eps,
                                 hidden_dropout_prob)
    
    def forward(self, hidden_states):
        self_fourier_outputs = self.fourier(hidden_states)
        fourier_output = self_fourier_outputs[0]
        intermediate_output = self.intermediate(fourier_output)
        layer_output = self.output(intermediate_output, fourier_output)
        
        return layer_output,


class FNetEncoder(Layer):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 layer_norm_eps,
                 hidden_dropout_prob,
                 hidden_act,
                 num_hidden_layers):
        super().__init__()
        self.layers = nn.LayerList([FNetLayer(hidden_size,
                                              intermediate_size,
                                              layer_norm_eps,
                                              hidden_dropout_prob,
                                              hidden_act) for _ in range(num_hidden_layers)])
        self.gradient_checkpointing = False
    
    def forward(self, hidden_states, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if return_dict:
            return {"last_hidden_state": hidden_states,
                    "all_hidden_states": all_hidden_states
                    }
        return hidden_states,


class FNetPooler(Layer):
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


class FNetEmbeddings(Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    
    def __init__(
            self,
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps,
            pad_token_id,
    ):
        super(FNetEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        # NOTE: This is the project layer and will be needed. The original code allows for different embedding and different model dimensions.
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", paddle.arange(max_position_embeddings).expand((1, -1)))
    
    def forward(
            self,
            input_ids,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.projection(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class FNetBasicFourierTransform(Layer):
    def __init__(self):
        super().__init__()
        self.fourier_transform = paddle.fft.fftn
    
    def forward(self, hidden_states):
        outputs = self.fourier_transform(hidden_states).real()
        return outputs,


class FNetFourierTransform(Layer):
    def __init__(self, hidden_size, layer_norm_eps):
        super().__init__()
        self.fourier_transform = FNetBasicFourierTransform()
        self.output = FNetBasicOutput(hidden_size, layer_norm_eps)
    
    def forward(self, hidden_states):
        self_outputs = self.fourier_transform(hidden_states)
        fourier_output = self.output(self_outputs[0], hidden_states)
        return fourier_output,


class FNetPredictionHeadTransform(Layer):
    def __init__(self, hidden_size, layer_norm_eps, hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, str):
            self.transform_act_fn = ACT2FN[hidden_act]
        else:
            self.transform_act_fn = hidden_act
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class FNetLMPredictionHead(Layer):
    def __init__(self, hidden_size, vocab_size, layer_norm_eps, hidden_act):
        super().__init__()
        self.transform = FNetPredictionHeadTransform(hidden_size, layer_norm_eps, hidden_act)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size)
        
        self.bias = self.create_parameter(
            [vocab_size],
            is_bias=True,
            default_initializer=nn.initializer.Constant(value=0)
        )
        self.decoder.bias = self.bias
    
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class FNetOnlyMLMHead(Layer):
    def __init__(self, hidden_size, vocab_size, layer_norm_eps, hidden_act):
        super().__init__()
        self.predictions = FNetLMPredictionHead(hidden_size, vocab_size, layer_norm_eps, hidden_act)
    
    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class FNetOnlyNSPHead(Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.seq_relationship = nn.Linear(hidden_size, 2)
    
    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class FNetPreTrainingHeads(Layer):
    def __init__(self, hidden_size, vocab_size, layer_norm_eps, hidden_act):
        super().__init__()
        self.predictions = FNetLMPredictionHead(hidden_size, vocab_size, layer_norm_eps, hidden_act)
        self.seq_relationship = nn.Linear(hidden_size, 2)
    
    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class FNetPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained FNet models. It provides FNet related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "fnet-base": {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 4,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "pad_token_id": 3,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        "fnet-large": {
            "vocab_size": 32000,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 4096,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 4,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "pad_token_id": 3,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "fnet-base": "https://bj.bcebos.com/paddlenlp/models/transformers/fnet/fnet-base/model_state.pdparams",
            "fnet-large": "https://bj.bcebos.com/paddlenlp/models/transformers/fnet/fnet-large/model_state.pdparams",
        }
    }
    base_model_prefix = "fnet"
    
    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, layer):
        # Initialize the weights.
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.fnet.config["initializer_range"],
                    shape=layer.weight.shape)
            )
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.fnet.config["initializer_range"],
                    shape=layer.weight.shape)
            )
            if layer._padding_idx is not None:
                layer.weight[layer._padding_idx].set_value(
                    paddle.zeros_like(layer.weight[layer._padding_idx])
                )
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.ones_like(layer.weight))


@register_base_model
class FNetModel(FNetPretrainedModel):
    """
    The model can behave as an encoder, following the architecture described in `FNet: Mixing Tokens with Fourier
    Transforms <https://arxiv.org/abs/2105.03824>`__ by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago
    Ontanon.

    Args:
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in `FNetModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `FNetModel`.
            Defaults to `32000`.
        hidden_size (int, optional):
            Dimensionality of the encoder layer and pooler layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `3072`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to `glue_new`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids`. Defaults to `4`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to `0.02`.
            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`BertPretrainedModel.init_weights()` for how weights are initialized in `ElectraModel`.
        layer_norm_eps(float, optional):
            The `epsilon` parameter used in :class:`paddle.nn.LayerNorm` for initializing layer normalization layers.
            A small value to the variance added to the normalization layer to prevent division by zero.
            Defaults to `1e-12`.
        pad_token_id (int, optional):
            The index of padding token in the token vocabulary. Defaults to `3`.
        add_pooling_layer(bool, optional):
            Whether or not to add the pooling layer. Defaults to `True`.
    """
    
    def __init__(
            self,
            vocab_size=32000,
            hidden_size=768,
            num_hidden_layers=12,
            intermediate_size=3072,
            hidden_act="gelu_new",
            hidden_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=4,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=3,
            bos_token_id=1,
            eos_token_id=2,
            add_pooling_layer=True
    ):
        super(FNetModel, self).__init__()
        self.initializer_range = initializer_range
        self.num_hidden_layers = num_hidden_layers
        self.embeddings = FNetEmbeddings(
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps,
            pad_token_id
        )
        self.encoder = FNetEncoder(
            hidden_size,
            intermediate_size,
            layer_norm_eps,
            hidden_dropout_prob,
            hidden_act,
            num_hidden_layers
        )
        self.pooler = FNetPooler(hidden_size) if add_pooling_layer else None
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
         The FNetModel forward method.

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
             inputs_embeds (Tensor, optional):
                If you want to control how to convert `inputs_ids` indices into associated vectors, you can
                pass an embedded representation directly instead of passing `inputs_ids`.
            output_hidden_states (bool, optional):
                Whether or not to return all hidden states. Default to `None`.
             return_dict (bool, optional):
                 Whether or not to return a dict instead of a plain tuple. Default to `None`.


         Returns:
             tuple or Dict: Returns tuple (`sequence_output`, `pooled_output`, `encoder_outputs[1:]`)
             or a dict with last_hidden_state`, `pooled_output`, `all_hidden_states`, fields.

             With the fields:

             - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and has a shape of [`batch_size, sequence_length, hidden_size`].

             - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and
                has a shape of [batch_size, hidden_size].

             - `last_hidden_state` (Tensor):
                The output of the last encoder layer, it is also the `sequence_output`.
                It's data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].

             - `all_hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `all_hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

         Example:
             .. code-block::

                 import paddle
                 from paddlenlp.transformers.fnet.modeling import FNetModel
                 from paddlenlp.transformers.fnet.tokenizer import FNetTokenizer

                 tokenizer = FNetTokenizer.from_pretrained('fnet-base')
                 model = FNetModel.from_pretrained('fnet-base')

                 inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                 inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                 output = model(**inputs)
         """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if token_type_ids is None:
            token_type_ids = paddle.zeros(shape=input_shape, dtype="int64")
        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]
        pooler_output = self.pooler(sequence_output) if self.pooler is not None else None
        
        if return_dict:
            return {"last_hidden_state": sequence_output,
                    "pooler_output": pooler_output,
                    "all_hidden_states": encoder_outputs["all_hidden_states"]
                    }
        return (sequence_output, pooler_output) + encoder_outputs[1:]


class FNetForSequenceClassification(FNetPretrainedModel):
    """
    FNet Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        fnet (:class:`FNetModel`):
            An instance of FNetModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.

    """
    
    def __init__(self, fnet, num_classes=2):
        super(FNetForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.fnet = fnet
        
        self.dropout = nn.Dropout(self.fnet.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.fnet.config["hidden_size"], num_classes)
        
        # Initialize weights and apply final processing
        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
         The FNetForSequenceClassification forward method.

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
             inputs_embeds (Tensor, optional):
                If you want to control how to convert `inputs_ids` indices into associated vectors, you can
                pass an embedded representation directly instead of passing `inputs_ids`.
            output_hidden_states (bool, optional):
                Whether or not to return all hidden states. Default to `None`.
             return_dict (bool, optional):
                 Whether or not to return a dict instead of a plain tuple. Default to `None`.


         Returns:
            Tensor or Dict: Returns tensor `logits`, or a dict with `logits`, `hidden_states`, `attentions` fields.

            With the fields:

            - `logits` (Tensor):
                A tensor of the input text classification logits.
                Shape as `[batch_size, num_classes]` and dtype as float32.

            - `hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

         Example:
             .. code-block::

                 import paddle
                 from paddlenlp.transformers.fnet.modeling import FNetForSequenceClassification
                 from paddlenlp.transformers.fnet.tokenizer import FNetTokenizer

                 tokenizer = FNetTokenizer.from_pretrained('fnet-base')
                 model = FNetModel.from_pretrained('fnet-base')

                 inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                 inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                 output = model(**inputs)
         """
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if return_dict:
            return {
                "logits": logits,
                "hidden_states": outputs["all_hidden_states"],
            }
        return logits


class FNetForPreTraining(FNetPretrainedModel):
    """
    FNet Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
     """
    
    def __init__(self, fnet):
        super().__init__()
        
        self.fnet = fnet
        self.cls = FNetPreTrainingHeads(
            self.fnet.config["hidden_size"],
            self.fnet.config["vocab_size"],
            self.fnet.config["layer_norm_eps"],
            self.fnet.config["hidden_act"])
        
        self.init_weights()
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    def get_input_embeddings(self):
        return self.fnet.embeddings.word_embeddings
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            next_sentence_label=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        The FNetForPretraining forward method.

        Args:
            input_ids (Tensor):
                See :class:`FNetModel`.
            token_type_ids (Tensor, optional):
                See :class:`FNetModel`.
            position_ids(Tensor, optional):
                See :class:`FNetModel`.
            labels (LongTensor of shape (batch_size, sequence_length), optional):
                Labels for computing the masked language modeling loss.
            inputs_embeds(Tensor, optional):
                See :class:`FNetModel`.
            next_sentence_labels(Tensor):
                The labels of the next sentence prediction task, the dimensionality of `next_sentence_labels`
                is equal to `seq_relation_labels`. Its data type should be int64 and
                its shape is [batch_size, 1]
            output_hidden_states (bool, optional):
                See :class:`FNetModel`.
            return_dict (bool, optional):
                See :class:`FNetModel`.

        Returns:
            tuple or Dict: Returns tuple (`prediction_scores`, `seq_relationship_score`) or a dict with
            `prediction_logits`, `seq_relationship_logits`,  `hidden_states` fields.
        """
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0] if not return_dict \
            else outputs["last_hidden_state"]
        pooled_output = outputs[1] if not return_dict \
            else outputs["pooler_output"]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        
        if return_dict:
            return {
                "prediction_logits": prediction_scores,
                "seq_relationship_logits": seq_relationship_score,
                "hidden_states": outputs["all_hidden_states"]
            }
        return prediction_scores, seq_relationship_score, outputs["all_hidden_states"]


class FNetForMaskedLM(FNetPretrainedModel):
    """
    FNet Model with a `masked language modeling` head on top.

    Args:
        fnet (:class:`FNetModel`):
            An instance of :class:`FNetModel`.

    """
    
    def __init__(self, fnet):
        super().__init__()
        
        self.fnet = fnet
        self.cls = FNetOnlyMLMHead(
            self.fnet.config["hidden_size"],
            self.fnet.config["vocab_size"],
            self.fnet.config["layer_norm_eps"],
            self.fnet.config["hidden_act"])
        
        self.init_weights()
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    def get_input_embeddings(self):
        return self.fnet.embeddings.word_embeddings
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            next_sentence_label=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        The FNetForMaskedLM forward method.

        Args:
            input_ids (Tensor):
                See :class:`FNetModel`.
            token_type_ids (Tensor, optional):
                See :class:`FNetModel`.
            position_ids(Tensor, optional):
                See :class:`FNetModel`.
            inputs_embeds(Tensor, optional):
                See :class:`FNetModel`.
            labels(Tensor, optional):
                See :class:`FNetForPreTraining`.
            next_sentence_label(Tensor, optional):
                See :class:`FNetForPreTraining`.
            output_hidden_states(Tensor, optional):
                See :class:`FNetModel`.
            return_dict(bool, optional):
                See :class:`FNetModel`.

        Returns:
            Tensor or Dict: Returns tensor `prediction_scores` or a dict with `prediction_logits`, `hidden_states` fields.

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                and its shape is [batch_size, sequence_length, vocab_size].

            - `hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].
        """
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] if not return_dict \
            else outputs["last_hidden_state"]
        prediction_scores = self.cls(sequence_output)
        
        if return_dict:
            return {
                "prediction_logits": prediction_scores,
                "hidden_states": outputs["all_hidden_states"]
            }
        return prediction_scores, outputs["all_hidden_states"]


class FNetForNextSentencePrediction(FNetPretrainedModel):
    """
    FNet Model with a `next sentence prediction` head on top.

    Args:
        fnet (:class:`FNetModel`):
            An instance of :class:`FNetModel`.

    """
    
    def __init__(self, fnet):
        super().__init__()
        
        self.fnet = fnet
        self.cls = FNetOnlyNSPHead(self.fnet.config["hidden_size"])
        
        self.init_weights()
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    def get_input_embeddings(self):
        return self.fnet.embeddings.word_embeddings
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            next_sentence_label=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1] if not return_dict \
            else outputs["pooler_output"]
        seq_relationship_score = self.cls(pooled_output)
        
        if return_dict:
            return {
                "seq_relationship_logits": seq_relationship_score,
                "hidden_states": outputs["all_hidden_states"]
            }
        return seq_relationship_score, outputs["all_hidden_states"]


class FNetForMultipleChoice(FNetPretrainedModel):
    """
    FNet Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like SWAG tasks .

    Args:
        fnet (:class:`FNetModel`):
            An instance of FNetModel.

    """
    
    def __init__(self, fnet):
        super(FNetForMultipleChoice, self).__init__()
        self.fnet = fnet
        self.dropout = nn.Dropout(self.fnet.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.fnet.config["hidden_size"], 1)
        
        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        input_ids = input_ids.reshape([-1, input_ids.shape[-1]]) \
            if input_ids is not None else None
        token_type_ids = token_type_ids.reshape([-1, token_type_ids.shape[-1]]) \
            if token_type_ids is not None else None
        position_ids = position_ids.reshape([-1, position_ids.shape[-1]]) \
            if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.reshape([-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1]])
            if inputs_embeds is not None
            else None
        )
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1] if not return_dict else outputs["pooler_output"]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape([-1, num_choices])
        if return_dict:
            return {
                "logits": reshaped_logits,
                "hidden_states": outputs["all_hidden_states"],
            }
        return reshaped_logits


class FNetForTokenClassification(FNetPretrainedModel):
    """
    FNet Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        fnet (:class:`FNetModel`):
            An instance of FNetModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
    """
    
    def __init__(self, fnet, num_classes=2):
        super(FNetForTokenClassification, self).__init__()
        self.fnet = fnet
        self.num_classes = num_classes
        self.dropout = nn.Dropout(self.fnet.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.fnet.config["hidden_size"], self.num_classes)
        
        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] if not return_dict else outputs["last_hidden_state"]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if return_dict:
            return {
                "logits": logits,
                "hidden_states": outputs["all_hidden_states"],
            }
        return logits


class FNetForQuestionAnswering(FNetPretrainedModel):
    """
    FNet Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        fnet (:class:`FNetModel`):
            An instance of FNetModel.
        num_labels (int):
            The number of labels.

    """
    
    def __init__(self, fnet, num_labels):
        super(FNetForQuestionAnswering, self).__init__()
        self.num_labels = num_labels
        self.fnet = fnet
        self.qa_outputs = nn.Linear(self.fnet.config["hidden_size"], num_labels)
        
        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] if not return_dict else outputs["last_hidden_state"]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = paddle.split(logits, num_or_sections=1, axis=-1)
        start_logits = start_logits.squeeze(axis=-1)
        end_logits = start_logits.squeeze(axis=-1)
        if return_dict:
            return {
                "start_logits": start_logits,
                "end_logits": end_logits,
                "hidden_states": outputs["all_hidden_states"],
            }
        return start_logits, end_logits
