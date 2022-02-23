# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Salesforce and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import CrossEntropyLoss, MSELoss
from .. import PretrainedModel, register_base_model

__all__ = [
    'CTRLModel', "CTRLLMHeadModel", 'CTRLForSequenceClassification',
    'SinusoidalPositionalEmbedding', 'CTRLForCausalLM'
]


class SinusoidalPositionalEmbedding(nn.Embedding):
    """
    This module produces sinusoidal positional embeddings of any length.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out):
        n_pos, dim = out.shape
        out.stop_gradient = True
        position_ids = paddle.arange(0, n_pos, dtype=out.dtype).unsqueeze(1)
        indices = paddle.arange(0, dim // 2, dtype=out.dtype).unsqueeze(0)

        indices = 10000.0**(-2 * indices / dim)
        embeddings = paddle.matmul(position_ids, indices)
        sentinel = dim // 2
        out[:, 0:sentinel] = paddle.sin(embeddings)
        out[:, sentinel:] = paddle.cos(embeddings)

        return out

    @paddle.no_grad()
    def forward(self, position_ids):
        return super().forward(position_ids)


def scaled_dot_product_attention(q, k, v, mask, attention_mask=None):
    # calculate attention
    matmul_qk = paddle.matmul(q, k, transpose_y=True)

    scaled_attention_logits = matmul_qk / np.sqrt(k.shape[-1])

    if mask is not None:
        nd, ns = scaled_attention_logits.shape[
            -2], scaled_attention_logits.shape[-1]
        scaled_attention_logits += mask[ns - nd:ns, :ns] * -1e4

    if attention_mask is not None:
        # Apply the attention mask
        scaled_attention_logits = scaled_attention_logits + attention_mask

    attention_weights = F.softmax(scaled_attention_logits, axis=-1)

    output = paddle.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    """

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.depth = hidden_size // self.num_heads

        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)

        self.dense = nn.Linear(hidden_size, hidden_size)

    def split_into_heads(self, x, batch_size):
        x = x.reshape([batch_size, -1, self.num_heads, self.depth])
        return x.transpose(perm=[0, 2, 1, 3])

    def forward(self,
                v,
                k,
                q,
                mask,
                layer_past=None,
                attention_mask=None,
                use_cache=False,
                output_attentions=False):
        batch_size = q.shape[0]

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)
        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            k = paddle.concat([past_key, k], axis=-2)
            v = paddle.concat([past_value, v], axis=-2)

        if use_cache is True:
            present = paddle.stack([k, v])
        else:
            present = (None, )

        scaled_attention, attn = scaled_dot_product_attention(q, k, v, mask,
                                                              attention_mask)
        scaled_attention = scaled_attention.transpose([0, 2, 1, 3])

        original_size_attention = scaled_attention.reshape(
            shape=[batch_size, -1, self.hidden_size])
        output = self.dense(original_size_attention)

        outputs = (output, present)
        if output_attentions:
            outputs = outputs + (attn, )
        return outputs


class EncoderLayer(nn.Layer):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 intermediate_size,
                 rate=0.1,
                 epsilon=1e-6):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(hidden_size, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
        self.layernorm1 = nn.LayerNorm(hidden_size, epsilon=epsilon)
        self.layernorm2 = nn.LayerNorm(hidden_size, epsilon=epsilon)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self,
                x,
                mask,
                layer_past=None,
                attention_mask=None,
                use_cache=False,
                output_attentions=False):
        normed = self.layernorm1(x)
        attn_outputs = self.multi_head_attention(
            normed,
            normed,
            normed,
            mask,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions, )
        attn_output = attn_outputs[0]
        attn_output = self.dropout1(attn_output)
        out1 = x + attn_output

        out2 = self.layernorm2(out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output)
        out2 = out1 + ffn_output

        outputs = (out2, ) + attn_outputs[1:]
        return outputs


class CTRLPreTrainedModel(PretrainedModel):
    """
    An abstract class for pretrained CTRL models. It provides CTRL related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    base_model_prefix = "ctrl"
    model_config_file = "model_config.json"

    pretrained_init_configuration = {
        "ctrl": {
            "tie_word_embeddings": True,
            "intermediate_size": 8192,
            "embd_pdrop": 0.1,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-06,
            "hidden_size": 1280,
            "num_attention_heads": 16,
            "num_hidden_layers": 48,
            "max_position_embeddings": 50000,
            "resid_pdrop": 0.1,
            "vocab_size": 246534,
            "pad_token_id": None
        },
        "sshleifer-tiny-ctrl": {
            "tie_word_embeddings": True,
            "intermediate_size": 2,
            "embd_pdrop": 0.1,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-06,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "max_position_embeddings": 50000,
            "resid_pdrop": 0.1,
            "vocab_size": 246534,
            "pad_token_id": None
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "ctrl":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ctrl/model_state.pdparams",
            "sshleifer-tiny-ctrl":
            "https://bj.bcebos.com/paddlenlp/models/transformers/sshleifer-tiny-ctrl/model_state.pdparams"
        }
    }

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else self.ctrl.config[
                        "initializer_range"],
                    shape=layer.weight.shape, ))
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else self.ctrl.config[
                        "initializer_range"],
                    shape=layer.weight.shape, ))
            if layer._padding_idx is not None:
                emb_weight = layer.weight.numpy()
                emb_weight[layer._padding_idx] = np.zeros_like(emb_weight[
                    layer._padding_idx])
                layer.weight.set_value(paddle.to_tensor(emb_weight))
        elif isinstance(layer, nn.LayerNorm):
            layer.weight.set_value(paddle.ones_like(layer.weight))
            layer.bias.set_value(paddle.zeros_like(layer.bias))


@register_base_model
class CTRLModel(CTRLPreTrainedModel):
    """
    The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in `CTRLModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `CTRLModel`.
            Defaults to `246534`.
        max_position_embeddings (int, optional):
            The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048 or 50000). 
            Defaults to `50000`.
        hidden_size (int, optional):
            Dimensionality of the embeddings and hidden states.
            Defaults to `1280`.
        intermediate_size (int, optional):
            Dimensionality of the inner dimension of the feed forward networks (FFN). 
            Defaults to `8192`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder.
            Defaults to `48`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `16`.
        resid_pdrop (float, optional):
            The dropout ratio for all fully connected layers in the encoder.
            Defaults to `0.1`.
        embd_pdrop (float, optional):
            The dropout ratio for the embeddings.
            Defaults to `0.1`.
        layer_norm_epsilon  (float, optional):
            The epsilon to use in the layer normalization layers. 
            Defaults to `1e-6`.
        tie_word_embeddings (bool, optional):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the model has a output word embedding layer.
            Defaults to `True`.
        pad_token_id (bool, optional):
            The id of the `padding` token.
            Defaults to `None`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer.
            Defaults to 0.02.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`CTRLPreTrainedModel._init_weights()` for how weights are initialized in `CTRLModel`.

    """

    def __init__(self,
                 vocab_size=246534,
                 max_position_embeddings=50000,
                 hidden_size=1280,
                 intermediate_size=8192,
                 num_hidden_layers=48,
                 num_attention_heads=16,
                 resid_pdrop=0.1,
                 embd_pdrop=0.1,
                 layer_norm_epsilon=1e-6,
                 tie_word_embeddings=True,
                 pad_token_id=None,
                 initializer_range=0.02):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_hidden_layers
        self.initializer_range = initializer_range

        self.pos_encoding = SinusoidalPositionalEmbedding(
            max_position_embeddings, self.hidden_size)

        self.w = nn.Embedding(vocab_size, hidden_size)

        self.dropout = nn.Dropout(embd_pdrop)
        self.h = nn.LayerList([
            EncoderLayer(hidden_size, num_attention_heads, intermediate_size,
                         resid_pdrop, layer_norm_epsilon)
            for _ in range(self.num_layers)
        ])
        self.layernorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.w

    def set_input_embeddings(self, new_embeddings):
        self.w = new_embeddings

    def forward(self,
                input_ids=None,
                cache=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False):
        r'''
        The CTRLModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            cache (Tuple[Tuple[Tensor]], optional):
                Contains pre-computed hidden-states (key and values in the attention blocks) 
                as computed by the model. Can be used to speed up sequential decoding. 
                The `input_ids` which have their past given to this model should not be 
                passed as input ids as they have already been computed.
                Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some 
                unwanted positions, usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others 
                have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `0.0` values and the others have `1.0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range `[0, type_vocab_size - 1]`.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            position_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected 
                in the range `[0, max_position_embeddings - 1]`.
                Shape as [batch_size, num_tokens] and dtype as int64. Defaults to `None`.
            use_cache (bool, optional):
                Whether or not to use cache. Defaults to `False`. If set to `True`, key value states 
                will be returned and can be used to speed up decoding.
            output_attentions (bool, optional):
                Whether or not to return the attentions tensors of all attention layers.
                Defaults to `False`.
            output_hidden_states (bool, optional):
                Whether or not to return the output of all hidden layers.
                Defaults to `False`.

        Returns:
            tuple: Returns tuple (`last_hidden_state`, `caches`, `hidden_states`, `attentions`)

            With the fields:

            - `last_hidden_state` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `caches` (tuple(tuple(Tensor), optional):
                returned when `use_cache=True` is passed.
                Tuple of `tuple(Tensor)` of length `num_hidden_layers`, with each tuple having 2 
                tensors of shape [batch_size, num_heads, sequence_length, embed_size_per_head] and float32 dtype.

            - `hidden_states` (tuple(Tensor), optional):
                returned when `output_hidden_states=True` is passed.
                Tuple of `Tensor` (one for the output of the embeddings + one for the output of 
                each layer). Each Tensor has a data type of float32 and its shape is 
                [batch_size, sequence_length, hidden_size].

            - `attentions` (tuple(Tensor), optional):
                returned when `output_attentions=True` is passed.
                Tuple of `Tensor` (one for each layer) of shape. Each Tensor has a data type of 
                float32 and its shape is [batch_size, num_heads, sequence_length, sequence_length].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import CTRLModel, CTRLTokenizer

                tokenizer = CTRLTokenizer.from_pretrained('ctrl')
                model = CTRLModel.from_pretrained('ctrl')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)

        '''

        seq_len = input_ids.shape[-1]
        input_ids = input_ids.reshape([-1, seq_len])
        batch_size = input_ids.shape[0]

        if cache is None:
            past_length = 0
            cache = tuple([None] * len(self.h))
        else:
            past_length = cache[0][0].shape[-2]

        if position_ids is None:
            position_ids = paddle.arange(past_length, seq_len + past_length)
            position_ids = position_ids.unsqueeze(0).reshape(
                shape=[-1, seq_len])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.reshape(shape=[batch_size, -1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze([1, 2])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.astype(
                dtype=paddle.get_default_dtype())  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(shape=[-1, seq_len])
            token_type_embeds = self.w(token_type_ids) * np.sqrt(
                self.hidden_size)
        else:
            token_type_embeds = 0.0

        inputs_embeds = self.w(input_ids) * np.sqrt(self.hidden_size)
        pos_embeds = self.pos_encoding(position_ids)

        hidden_states = inputs_embeds + pos_embeds + token_type_embeds

        hidden_states = self.dropout(hidden_states)
        mask = paddle.triu(
            paddle.ones(shape=[seq_len + past_length, seq_len + past_length]),
            1)

        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, (h, layer_past) in enumerate(zip(self.h, cache)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )
            outputs = h(
                hidden_states,
                mask,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions, )
            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present, )

            if output_attentions:
                all_attentions += (outputs[2], )

        hidden_states = self.layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        return tuple(
            v
            for v in
            [hidden_states, presents, all_hidden_states, all_attentions]
            if v is not None)


class CTRLLMHeadModel(CTRLPreTrainedModel):
    """
    The CTRL Model transformer with a language modeling head on top (linear 
    layer with weights tied to the input embeddings).

    Args:
        ctrl (:class:`CTRLModel`):
            An instance of :class:`CTRLModel`.

    """

    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        if self.ctrl.config["tie_word_embeddings"]:
            self.lm_head = self.ctrl.w
            self.lm_head_bias = self.create_parameter(
                shape=[self.ctrl.config["vocab_size"]],
                dtype=self.lm_head.weight.dtype,
                is_bias=True, )
        else:
            self.lm_head = nn.Linear(self.ctrl.config["hidden_size"],
                                     self.ctrl.config["vocab_size"])

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      use_cache=False,
                                      cache=None,
                                      **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {"input_ids": input_ids, "use_cache": use_cache, "cache": cache}

    def forward(self,
                input_ids=None,
                cache=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`CTRLModel`.
            cache (Tensor, optional):
                See :class:`CTRLModel`.
            attention_mask (Tensor, optional):
                See :class:`CTRLModel`.
            token_type_ids (Tensor, optional):
                See :class:`CTRLModel`.
            position_ids (Tensor, optional):
                See :class:`CTRLModel`.
            labels (Tensor, optional):
                Labels for language modeling. Note that the labels **are shifted** 
                inside the model, i.e. you can set `labels = input_ids` Indices are 
                selected in `[-100, 0, ..., vocab_size]` All labels set to `-100` are 
                ignored (masked), the loss is only computed for labels in `[0, ..., vocab_size]`.
                Shape is [batch_size, sequence_length] and dtype is int64.
            use_cache (bool, optional):
                See :class:`CTRLModel`.
            output_attentions (bool, optional):
                See :class:`CTRLModel`.
            output_hidden_states (bool, optional):
                See :class:`CTRLModel`.

        Returns:
            tuple: Returns tuple `(loss, logits, caches, hidden_states, attentions)`.
            With the fields:

            - `loss` (Tensor):
                returned when `labels` is provided.
                Language modeling loss (for next-token prediction).
                It's data type should be float32 and its shape is [1,].

            - `logits` (Tensor):
                Prediction scores of the language modeling head (scores for each vocabulary 
                token before SoftMax).
                It's data type should be float32 and 
                its shape is [batch_size, sequence_length, vocab_size].

            - `caches` (tuple(tuple(Tensor), optional):
                See :class:`CTRLModel`.

            - `hidden_states` (tuple(Tensor), optional):
                See :class:`CTRLModel`.

            - `attentions` (tuple(Tensor), optional):
                See :class:`CTRLModel`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import CTRLLMHeadModel, CTRLTokenizer

                tokenizer = CTRLTokenizer.from_pretrained('ctrl')
                model = CTRLLMHeadModel.from_pretrained('ctrl')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs, labels=inputs["input_ids"])

                loss = output[0]
                logits = output[1]

        """

        ctrl_outputs = self.ctrl(
            input_ids,
            cache=cache,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)

        hidden_states = ctrl_outputs[0]

        if self.ctrl.config["tie_word_embeddings"]:
            lm_logits = (paddle.matmul(
                hidden_states, self.lm_head.weight, transpose_y=True) +
                         self.lm_head_bias)
        else:
            lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[:, :-1]
            shift_labels = labels[:, 1:]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape([-1, shift_logits.shape[-1]]),
                shift_labels.flatten(), )

        output = (lm_logits, ) + ctrl_outputs[1:]
        return ((loss, ) + output) if loss is not None else output


class CTRLForSequenceClassification(CTRLPreTrainedModel):
    """
    The CTRL Model transformer with a sequence classification head on top (linear layer).
    `CTRLForSequenceClassification` uses the last token in order to do the classification, 
    as other causal models (e.g. GPT-2) do. Since it does classification on the last token, 
    it requires to know the position of the last token. If a `pad_token_id` is defined in the 
    configuration, it finds the last token that is not a padding token in each row. If no 
    `pad_token_id` is defined, it simply takes the last value in each row of the batch. 

    Args:
        ctrl (:class:`CTRLModel`):
            An instance of :class:`CTRLModel`.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of CTRL.
            If None, use the same value as `hidden_dropout_prob` of `CTRLModel`
            instance `ctrl`. Defaults to None.

    """

    def __init__(self, ctrl, num_classes=2, dropout=None):
        super().__init__()
        self.num_classes = num_classes
        self.ctrl = ctrl
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ctrl.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(
            self.ctrl.config["hidden_size"], num_classes, bias_attr=False)

        self.init_weights()

    def forward(self,
                input_ids=None,
                cache=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`CTRLModel`.
            cache (Tensor, optional):
                See :class:`CTRLModel`.
            attention_mask (Tensor, optional):
                See :class:`CTRLModel`.
            token_type_ids (Tensor, optional):
                See :class:`CTRLModel`.
            position_ids (Tensor, optional):
                See :class:`CTRLModel`.
            labels (Tensor, optional):
                Labels for computing the sequence classification/regression loss. 
                Indices should be in `[0, ...,num_classes - 1]`. If `num_classes == 1` 
                a regression loss is computed (Mean-Square loss), If `num_classes > 1` 
                a classification loss is computed (Cross-Entropy).
                Shape is [batch_size,] and dtype is int64.
            use_cache (bool, optional):
                See :class:`CTRLModel`.
            output_attentions (bool, optional):
                See :class:`CTRLModel`.
            output_hidden_states (bool, optional):
                See :class:`CTRLModel`.

        Returns:
            tuple: Returns tuple `(loss, logits, caches, hidden_states, attentions)`.
            With the fields:

            - `loss` (Tensor):
                returned when `labels` is provided.
                Language modeling loss (for next-token prediction).
                It's data type should be float32 and its shape is [1,].

            - `logits` (Tensor):
                Prediction scores of the language modeling head (scores for each vocabulary 
                token before SoftMax).
                It's data type should be float32 and its shape is [batch_size, num_classes].

            - `caches` (tuple(tuple(Tensor), optional):
                See :class:`CTRLModel`.

            - `hidden_states` (tuple(Tensor), optional):
                See :class:`CTRLModel`.

            - `attentions` (tuple(Tensor), optional):
                See :class:`CTRLModel`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import CTRLForSequenceClassification, CTRLTokenizer

                tokenizer = CTRLTokenizer.from_pretrained('ctrl')
                model = CTRLForSequenceClassification.from_pretrained('ctrl', pad_token_id=0)

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs, labels=paddle.to_tensor([1]))

                loss = output[0]
                logits = output[1]

        """
        ctrl_outputs = self.ctrl(
            input_ids,
            cache=cache,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)

        hidden_states = ctrl_outputs[0]
        logits = self.classifier(hidden_states)
        batch_size = input_ids.shape[0]

        assert (
            self.ctrl.config["pad_token_id"] is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."

        if self.ctrl.config["pad_token_id"] is None:
            sequence_lengths = -1
        else:
            sequence_lengths = paddle.not_equal(
                input_ids, self.ctrl.config["pad_token_id"]
                .astype(paddle.int64).sum(-1) - 1)

        pooled_logits = logits.gather_nd(
            paddle.stack(
                [paddle.arange(batch_size), sequence_lengths], axis=-1))

        loss = None
        if labels is not None:
            if self.num_classes == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(pooled_logits.flatten(),
                                labels.astype(pooled_logits.dtype).flatten())
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.reshape([-1, self.num_classes]),
                    labels.flatten())

        output = (pooled_logits, ) + ctrl_outputs[1:]
        return ((loss, ) + output) if loss is not None else output


CTRLForCausalLM = CTRLLMHeadModel
