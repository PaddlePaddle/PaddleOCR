# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import paddle
from paddlenlp.transformers.roberta.modeling import (
    RobertaForMaskedLM,
    RobertaConfig,
    MaskedLMOutput,
)
from paddle import Tensor


class RobertHead(RobertaForMaskedLM):
    """
    Roberta Model with a `masked language modeling` head on top.

    Args:
        bert (:class:RobertaModel`):
            An instance of :class:`RobertaModel`.

    """

    def __init__(self, **config):
        vocab_size = config["decoder_args"].pop("vocab_size")
        hidden_size = config["decoder_args"].pop("hidden_size")
        num_hidden_layers = config["decoder_args"].pop("num_hidden_layers")
        num_attention_heads = config["decoder_args"].pop("num_attention_heads")
        intermediate_size = config["decoder_args"].pop("intermediate_size")
        hidden_act = config["decoder_args"].pop("hidden_act")
        hidden_dropout_prob = config["decoder_args"].pop("hidden_dropout_prob")
        attention_probs_dropout_prob = config["decoder_args"].pop(
            "attention_probs_dropout_prob"
        )
        max_position_embeddings = config["decoder_args"].pop("max_position_embeddings")
        type_vocab_size = config["decoder_args"].pop("type_vocab_size")
        initializer_range = config["decoder_args"].pop("initializer_range")
        pad_token_id = config["decoder_args"].pop("pad_token_id")
        layer_norm_eps = config["decoder_args"].pop("layer_norm_eps", 1e-5)
        cls_token_id = config["decoder_args"].pop("cls_token_id", None)
        robert_config = RobertaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            pad_token_id=pad_token_id,
            layer_norm_eps=layer_norm_eps,
            cls_token_id=cls_token_id,
            **config,
        )
        super().__init__(robert_config)

    # forward for export
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        r"""


        Args:
            input_ids (Tensor):
                See :class:`RobertaModel`.
            token_type_ids (Tensor, optional):
                Segment token indices to indicate first and second portions of the inputs.
                Indices can be either 0 or 1:

                - 0 corresponds to a **sentence A** token,
                - 1 corresponds to a **sentence B** token.

                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
                Defaults to None, which means no segment embeddings is added to token embeddings.
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings.
                Selected in the range ``[0, max_position_embeddings - 1]``.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.
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
            inputs_embeds (Tensor, optional):
                If you want to control how to convert `inputs_ids` indices into associated vectors, you can
                pass an embedded representation directly instead of passing `inputs_ids`.
            labels (Tensor of shape `(batch_size, sequence_length)`, optional):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., vocab_size]`
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.MaskedLMOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.MaskedLMOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddlenlp.transformers.model_outputs.MaskedLMOutput`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RobertaForMaskedLM, RobertaTokenizer

                tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
                model = RobertaForMaskedLM.from_pretrained('roberta-wwm-ext')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
                print(logits.shape)
                # [1, 13, 30522]
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        input_ids = input_ids.astype(paddle.int64)
        outputs = self.roberta(
            input_ids,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = paddle.nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.reshape((-1, prediction_scores.shape[-1])),
                labels.reshape((-1,)),
            )
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output)
                if masked_lm_loss is not None
                else (output[0] if len(output) == 1 else output)
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
