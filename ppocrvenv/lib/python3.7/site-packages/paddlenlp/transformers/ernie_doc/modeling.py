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

from ..attention_utils import _convert_param_attr_to_list
from .. import PretrainedModel, register_base_model

__all__ = [
    'ErnieDocModel',
    'ErnieDocPretrainedModel',
    'ErnieDocForSequenceClassification',
    'ErnieDocForTokenClassification',
    'ErnieDocForQuestionAnswering',
]


class PointwiseFFN(nn.Layer):
    def __init__(self,
                 d_inner_hid,
                 d_hid,
                 dropout_rate,
                 hidden_act,
                 weight_attr=None,
                 bias_attr=None):
        super(PointwiseFFN, self).__init__()
        self.linear1 = nn.Linear(
            d_hid, d_inner_hid, weight_attr, bias_attr=bias_attr)
        self.dropout = nn.Dropout(dropout_rate, mode="upscale_in_train")
        self.linear2 = nn.Linear(
            d_inner_hid, d_hid, weight_attr, bias_attr=bias_attr)
        self.activation = getattr(F, hidden_act)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class MultiHeadAttention(nn.Layer):
    def __init__(self,
                 d_key,
                 d_value,
                 d_model,
                 n_head=1,
                 r_w_bias=None,
                 r_r_bias=None,
                 r_t_bias=None,
                 dropout_rate=0.,
                 weight_attr=None,
                 bias_attr=None):
        super(MultiHeadAttention, self).__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.n_head = n_head

        assert d_key * n_head == d_model, "d_model must be divisible by n_head"

        self.q_proj = nn.Linear(
            d_model,
            d_key * n_head,
            weight_attr=weight_attr,
            bias_attr=bias_attr)
        self.k_proj = nn.Linear(
            d_model,
            d_key * n_head,
            weight_attr=weight_attr,
            bias_attr=bias_attr)
        self.v_proj = nn.Linear(
            d_model,
            d_value * n_head,
            weight_attr=weight_attr,
            bias_attr=bias_attr)
        self.r_proj = nn.Linear(
            d_model,
            d_key * n_head,
            weight_attr=weight_attr,
            bias_attr=bias_attr)
        self.t_proj = nn.Linear(
            d_model,
            d_key * n_head,
            weight_attr=weight_attr,
            bias_attr=bias_attr)
        self.out_proj = nn.Linear(
            d_model, d_model, weight_attr=weight_attr, bias_attr=bias_attr)
        self.r_w_bias = r_w_bias
        self.r_r_bias = r_r_bias
        self.r_t_bias = r_t_bias
        self.dropout = nn.Dropout(
            dropout_rate, mode="upscale_in_train") if dropout_rate else None

    def __compute_qkv(self, queries, keys, values, rel_pos, rel_task):
        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)
        r = self.r_proj(rel_pos)
        t = self.t_proj(rel_task)
        return q, k, v, r, t

    def __split_heads(self, x, d_model, n_head):
        # x shape: [B, T, H]
        x = x.reshape(shape=[0, 0, n_head, d_model // n_head])
        # shape: [B, N, T, HH]
        return paddle.transpose(x=x, perm=[0, 2, 1, 3])

    def __rel_shift(self, x, klen=-1):
        """
        To perform relative attention, it should relatively shift the attention score matrix
        See more details on: https://github.com/kimiyoung/transformer-xl/issues/8#issuecomment-454458852        
        """
        # input shape: [B, N, T, 2 * T + M]
        x_shape = x.shape
        x = x.reshape([x_shape[0], x_shape[1], x_shape[3], x_shape[2]])
        x = x[:, :, 1:, :]
        x = x.reshape([x_shape[0], x_shape[1], x_shape[2], x_shape[3] - 1])
        # output shape: [B, N, T, T + M]
        return x[:, :, :, :klen]

    def __scaled_dot_product_attention(self, q, k, v, r, t, attn_mask):
        q_w, q_r, q_t = q
        score_w = paddle.matmul(q_w, k, transpose_y=True)
        score_r = paddle.matmul(q_r, r, transpose_y=True)
        score_r = self.__rel_shift(score_r, k.shape[2])
        score_t = paddle.matmul(q_t, t, transpose_y=True)
        score = score_w + score_r + score_t
        score = score * (self.d_key**-0.5)
        if attn_mask is not None:
            score += attn_mask
        weights = F.softmax(score)
        if self.dropout:
            weights = self.dropout(weights)
        out = paddle.matmul(weights, v)
        return out

    def __combine_heads(self, x):
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")
        # x shape: [B, N, T, HH]
        x = paddle.transpose(x, [0, 2, 1, 3])
        # target shape:[B, T, H]
        return x.reshape([0, 0, x.shape[2] * x.shape[3]])

    def forward(self, queries, keys, values, rel_pos, rel_task, memory,
                attn_mask):
        if memory is not None and len(memory.shape) > 1:
            cat = paddle.concat([memory, queries], 1)
        else:
            cat = queries
        keys, values = cat, cat

        if not (len(queries.shape) == len(keys.shape) == len(values.shape) \
            == len(rel_pos.shape) == len(rel_task.shape)== 3):
            raise ValueError(
                "Inputs: quries, keys, values, rel_pos and rel_task should all be 3-D tensors."
            )

        q, k, v, r, t = self.__compute_qkv(queries, keys, values, rel_pos,
                                           rel_task)
        q_w, q_r, q_t = list(
            map(lambda x: q + x.unsqueeze([0, 1]),
                [self.r_w_bias, self.r_r_bias, self.r_t_bias]))
        q_w, q_r, q_t = list(
            map(lambda x: self.__split_heads(x, self.d_model, self.n_head),
                [q_w, q_r, q_t]))
        k, v, r, t = list(
            map(lambda x: self.__split_heads(x, self.d_model, self.n_head),
                [k, v, r, t]))

        ctx_multiheads = self.__scaled_dot_product_attention([q_w, q_r, q_t], \
                                    k, v, r, t, attn_mask)

        out = self.__combine_heads(ctx_multiheads)
        out = self.out_proj(out)
        return out


class ErnieDocEncoderLayer(nn.Layer):
    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 hidden_act,
                 normalize_before=False,
                 epsilon=1e-5,
                 rel_pos_params_sharing=False,
                 r_w_bias=None,
                 r_r_bias=None,
                 r_t_bias=None,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3
        super(ErnieDocEncoderLayer, self).__init__()
        if not rel_pos_params_sharing:
            r_w_bias, r_r_bias, r_t_bias = \
                list(map(lambda x: self.create_parameter(
                    shape=[n_head * d_key], dtype="float32"),
                    ["r_w_bias", "r_r_bias", "r_t_bias"]))

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)
        self.attn = MultiHeadAttention(
            d_key,
            d_value,
            d_model,
            n_head,
            r_w_bias,
            r_r_bias,
            r_t_bias,
            attention_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0], )
        self.ffn = PointwiseFFN(
            d_inner_hid,
            d_model,
            relu_dropout,
            hidden_act,
            weight_attr=weight_attrs[1],
            bias_attr=bias_attrs[1])
        self.norm1 = nn.LayerNorm(d_model, epsilon=epsilon)
        self.norm2 = nn.LayerNorm(d_model, epsilon=epsilon)
        self.dropout1 = nn.Dropout(
            prepostprocess_dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(
            prepostprocess_dropout, mode="upscale_in_train")
        self.d_model = d_model
        self.epsilon = epsilon
        self.normalize_before = normalize_before

    def forward(self, enc_input, memory, rel_pos, rel_task, attn_mask):
        residual = enc_input
        if self.normalize_before:
            enc_input = self.norm1(enc_input)
        attn_output = self.attn(enc_input, enc_input, enc_input, rel_pos,
                                rel_task, memory, attn_mask)
        attn_output = residual + self.dropout1(attn_output)
        if not self.normalize_before:
            attn_output = self.norm1(attn_output)
        residual = attn_output
        if self.normalize_before:
            attn_output = self.norm2(attn_output)
        ffn_output = self.ffn(attn_output)
        output = residual + self.dropout2(ffn_output)
        if not self.normalize_before:
            output = self.norm2(output)
        return output


class ErnieDocEncoder(nn.Layer):
    def __init__(self, num_layers, encoder_layer, mem_len):
        super(ErnieDocEncoder, self).__init__()
        self.layers = nn.LayerList([(
            encoder_layer
            if i == 0 else type(encoder_layer)(**encoder_layer._config))
                                    for i in range(num_layers)])
        self.num_layers = num_layers
        self.normalize_before = self.layers[0].normalize_before
        self.mem_len = mem_len

    def _cache_mem(self, curr_out, prev_mem):
        if self.mem_len is None or self.mem_len == 0:
            return None
        if prev_mem is None:
            new_mem = curr[:, -self.mem_len:, :]
        else:
            new_mem = paddle.concat([prev_mem, curr_out],
                                    1)[:, -self.mem_len:, :]
        new_mem.stop_gradient = True
        return new_mem

    def forward(self, enc_input, memories, rel_pos, rel_task, attn_mask):
        # no need to normalize enc_input, cause it's already normalized outside.
        new_mem = []
        for i, encoder_layer in enumerate(self.layers):
            enc_input = encoder_layer(enc_input, memories[i], rel_pos, rel_task,
                                      attn_mask)
            new_mem += [self._cache_mem(enc_input, memories[i])]
            # free the old memories explicitly to save gpu memory
            memories[i] = None
        return enc_input, new_mem


class ErnieDocPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained ErnieDoc models. It provides ErnieDoc related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading
    and loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "ernie-doc-base-en": {
            "attention_dropout_prob": 0.0,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "relu_dropout": 0.0,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "task_type_vocab_size": 3,
            "vocab_size": 50265,
            "memory_len": 128,
            "epsilon": 1e-12,
            "pad_token_id": 1
        },
        "ernie-doc-base-zh": {
            "attention_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "relu_dropout": 0.0,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "task_type_vocab_size": 3,
            "vocab_size": 28000,
            "memory_len": 128,
            "epsilon": 1e-12,
            "pad_token_id": 0
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "ernie-doc-base-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-doc-base-en/ernie-doc-base-en.pdparams",
            "ernie-doc-base-zh":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-doc-base-zh/ernie-doc-base-zh.pdparams",
        }
    }
    base_model_prefix = "ernie_doc"

    def init_weights(self, layer):
        # Initialization hook
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.ernie_doc.config["initializer_range"],
                        shape=layer.weight.shape))


class ErnieDocEmbeddings(nn.Layer):
    def __init__(self,
                 vocab_size,
                 d_model,
                 hidden_dropout_prob,
                 memory_len,
                 max_position_embeddings=512,
                 type_vocab_size=3,
                 padding_idx=0):
        super(ErnieDocEmbeddings, self).__init__()
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_position_embeddings * 2 + memory_len,
                                    d_model)
        self.token_type_emb = nn.Embedding(type_vocab_size, d_model)
        self.memory_len = memory_len
        self.dropouts = nn.LayerList(
            [nn.Dropout(hidden_dropout_prob) for i in range(3)])
        self.norms = nn.LayerList([nn.LayerNorm(d_model) for i in range(3)])

    def forward(self, input_ids, token_type_ids, position_ids):
        # input_embeddings: [B, T, H]
        input_embeddings = self.word_emb(input_ids.squeeze(-1))
        # position_embeddings: [B, 2 * T + M, H]
        position_embeddings = self.pos_emb(position_ids.squeeze(-1))

        batch_size = input_ids.shape[0]
        token_type_ids = paddle.concat(
            [
                paddle.zeros(
                    shape=[batch_size, self.memory_len, 1], dtype="int64") +
                token_type_ids[0, 0, 0], token_type_ids
            ],
            axis=1)
        token_type_ids.stop_gradient = True
        # token_type_embeddings: [B, M + T, H]
        token_type_embeddings = self.token_type_emb(token_type_ids.squeeze(-1))
        embs = [input_embeddings, position_embeddings, token_type_embeddings]
        for i in range(len(embs)):
            embs[i] = self.dropouts[i](self.norms[i](embs[i]))
        return embs


class ErnieDocPooler(nn.Layer):
    """
    get pool output
    """

    def __init__(self, hidden_size, cls_token_idx=-1):
        super(ErnieDocPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.cls_token_idx = cls_token_idx

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the last token.
        cls_token_tensor = hidden_states[:, self.cls_token_idx]
        pooled_output = self.dense(cls_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@register_base_model
class ErnieDocModel(ErnieDocPretrainedModel):
    """
    The bare ERNIE-Doc Model outputting raw hidden-states.
    
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    
    This model is also a `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        num_hidden_layers (int):
            The number of hidden layers in the Transformer encoder.
        num_attention_heads (int):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_size (int):
            Dimensionality of the embedding layers, encoder layers and pooler layer.
        hidden_dropout_prob (int):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_dropout_prob (int):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
        relu_dropout (int):
            The dropout probability of FFN.
        hidden_act (str):
            The non-linear activation function of FFN.
        memory_len (int):
            The number of tokens to cache. If not 0, the last `memory_len` hidden states
            in each layer will be cached into memory.
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `ErnieDocModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `ErnieDocModel`.
        max_position_embeddings (int):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        task_type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids`. Defaults to `3`.
        normalize_before (bool, optional):
            Indicate whether to put layer normalization into preprocessing of MHA and FFN sub-layers. 
            If True, pre-process is layer normalization and post-precess includes dropout, 
            residual connection. Otherwise, no pre-process and post-precess includes dropout, 
            residual connection, layer normalization. Defaults to `False`.
        epsilon (float, optional):
            The `epsilon` parameter used in :class:`paddle.nn.LayerNorm` for
            initializing layer normalization layers. Defaults to `1e-5`.
        rel_pos_params_sharing (bool, optional):
            Whether to share the relative position parameters.
            Defaults to `False`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer for initializing all weight matrices.
            Defaults to `0.02`.
        pad_token_id (int, optional):
            The token id of [PAD] token whose parameters won't be updated when training.
            Defaults to `0`.
        cls_token_idx (int, optional):
            The token id of [CLS] token. Defaults to `-1`.
    """

    def __init__(self,
                 num_hidden_layers,
                 num_attention_heads,
                 hidden_size,
                 hidden_dropout_prob,
                 attention_dropout_prob,
                 relu_dropout,
                 hidden_act,
                 memory_len,
                 vocab_size,
                 max_position_embeddings,
                 task_type_vocab_size=3,
                 normalize_before=False,
                 epsilon=1e-5,
                 rel_pos_params_sharing=False,
                 initializer_range=0.02,
                 pad_token_id=0,
                 cls_token_idx=-1):
        super(ErnieDocModel, self).__init__()

        r_w_bias, r_r_bias, r_t_bias = None, None, None
        if rel_pos_params_sharing:
            r_w_bias, r_r_bias, r_t_bias = \
                list(map(lambda x: self.create_parameter(
                    shape=[num_attention_heads * d_key], dtype="float32"),
                    ["r_w_bias", "r_r_bias", "r_t_bias"]))
        d_key = hidden_size // num_attention_heads
        d_value = hidden_size // num_attention_heads
        d_inner_hid = hidden_size * 4
        encoder_layer = ErnieDocEncoderLayer(
            num_attention_heads,
            d_key,
            d_value,
            hidden_size,
            d_inner_hid,
            hidden_dropout_prob,
            attention_dropout_prob,
            relu_dropout,
            hidden_act,
            normalize_before=normalize_before,
            epsilon=epsilon,
            rel_pos_params_sharing=rel_pos_params_sharing,
            r_w_bias=r_w_bias,
            r_r_bias=r_r_bias,
            r_t_bias=r_t_bias)
        self.n_head = num_attention_heads
        self.d_model = hidden_size
        self.memory_len = memory_len
        self.encoder = ErnieDocEncoder(num_hidden_layers, encoder_layer,
                                       memory_len)
        self.pad_token_id = pad_token_id
        self.embeddings = ErnieDocEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob, memory_len,
            max_position_embeddings, task_type_vocab_size, pad_token_id)
        self.pooler = ErnieDocPooler(hidden_size, cls_token_idx)

    def _create_n_head_attn_mask(self, attn_mask, batch_size):
        # attn_mask shape: [B, T, 1]
        # concat an data_mask, shape: [B, M + T, 1]
        data_mask = paddle.concat(
            [
                paddle.ones(
                    shape=[batch_size, self.memory_len, 1],
                    dtype=attn_mask.dtype), attn_mask
            ],
            axis=1)
        data_mask.stop_gradient = True
        # create a self_attn_mask, shape: [B, T, M + T]
        self_attn_mask = paddle.matmul(attn_mask, data_mask, transpose_y=True)
        self_attn_mask = (self_attn_mask - 1) * 1e8
        n_head_self_attn_mask = paddle.stack(
            [self_attn_mask] * self.n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True
        return n_head_self_attn_mask

    def forward(self, input_ids, memories, token_type_ids, position_ids,
                attn_mask):
        r"""
        The ErnieDocModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length, 1].
            memories (List[Tensor]):
                A list of length `n_layers` with each Tensor being a pre-computed hidden-state for each layer.
                Each Tensor has a dtype `float32` and a shape of [batch_size, sequence_length, hidden_size].
            token_type_ids (Tensor):
                Segment token indices to indicate first and second portions of the inputs.
                Indices can be either 0 or 1:

                - 0 corresponds to a **sentence A** token,
                - 1 corresponds to a **sentence B** token.

                It's data type should be `int64` and has a shape of [batch_size, sequence_length, 1].
                Defaults to None, which means no segment embeddings is added to token embeddings.
            position_ids (Tensor):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                config.max_position_embeddings - 1]``. Shape as `(batch_sie, num_tokens)` and dtype as `int32` or `int64`.
            attn_mask (Tensor):
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

        Returns:
            tuple : Returns tuple (``encoder_output``, ``pooled_output``, ``new_mem``).

            With the fields:

            - `encoder_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].

            - `new_mem` (List[Tensor]):
                A list of pre-computed hidden-states. The length of the list is `n_layers`.
                Each element in the list is a Tensor with dtype `float32` and shape as [batch_size, memory_length, hidden_size].

        Example:
            .. code-block::

                import numpy as np
                import paddle
                from paddlenlp.transformers import ErnieDocModel
                from paddlenlp.transformers import ErnieDocTokenizer
                
                def get_related_pos(insts, seq_len, memory_len=128):
                    beg = seq_len + seq_len + memory_len
                    r_position = [list(range(beg - 1, seq_len - 1, -1)) + \
                                list(range(0, seq_len)) for i in range(len(insts))]
                    return np.array(r_position).astype('int64').reshape([len(insts), beg, 1])
                    
                tokenizer = ErnieDocTokenizer.from_pretrained('ernie-doc-base-zh')
                model = ErnieDocModel.from_pretrained('ernie-doc-base-zh')

                inputs = tokenizer("欢迎使用百度飞桨！")
                inputs = {k:paddle.to_tensor([v + [0] * (128-len(v))]).unsqueeze(-1) for (k, v) in inputs.items()}
                
                memories = [paddle.zeros([1, 128, 768], dtype="float32") for _ in range(12)]
                position_ids = paddle.to_tensor(get_related_pos(inputs['input_ids'], 128, 128))
                attn_mask = paddle.ones([1, 128, 1])

                inputs['memories'] = memories
                inputs['position_ids'] = position_ids
                inputs['attn_mask'] = attn_mask

                outputs = model(**inputs)

                encoder_output = outputs[0]
                pooled_output = outputs[1]
                new_mem = outputs[2]

        """
        input_embeddings, position_embeddings, token_embeddings = \
            self.embeddings(input_ids, token_type_ids, position_ids)

        batch_size = input_embeddings.shape[0]
        # [B, N, T, M + T]
        n_head_self_attn_mask = self._create_n_head_attn_mask(attn_mask,
                                                              batch_size)
        # memories contains n_layer memory whose shape is [B, M, H]
        encoder_output, new_mem = self.encoder(
            enc_input=input_embeddings,
            memories=memories,
            rel_pos=position_embeddings,
            rel_task=token_embeddings,
            attn_mask=n_head_self_attn_mask)
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output, new_mem


class ErnieDocForSequenceClassification(ErnieDocPretrainedModel):
    """
    ErnieDoc Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        ernie_doc (:class:`ErnieDocModel`):
            An instance of :class:`ErnieDocModel`.
        num_classes (int):
            The number of classes.
        dropout (float, optional)
            The dropout ratio of last output. Default to `0.1`.
    """

    def __init__(self, ernie_doc, num_classes, dropout=0.1):
        super(ErnieDocForSequenceClassification, self).__init__()
        self.ernie_doc = ernie_doc
        self.linear = nn.Linear(self.ernie_doc.config["hidden_size"],
                                num_classes)
        self.dropout = nn.Dropout(dropout, mode="upscale_in_train")
        self.apply(self.init_weights)

    def forward(self, input_ids, memories, token_type_ids, position_ids,
                attn_mask):
        r"""
        The ErnieDocForSequenceClassification forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                See :class:`ErnieDocModel`.
            memories (List[Tensor]):
                See :class:`ErnieDocModel`.
            token_type_ids (Tensor):
                See :class:`ErnieDocModel`.
            position_ids (Tensor):
                See :class:`ErnieDocModel`.
            attn_mask (Tensor):
                See :class:`ErnieDocModel`.

        Returns:
            tuple : Returns tuple (`logits`, `mem`).

            With the fields:

            - `logits` (Tensor):
                A tensor containing the [CLS] of hidden-states of the model at the output of last layer.
                Each Tensor has a data type of `float32` and has a shape of [batch_size, num_classes].

            - `mem` (List[Tensor]):
                A list of pre-computed hidden-states. The length of the list is `n_layers`.
                Each element in the list is a Tensor with dtype `float32` and has a shape of
                [batch_size, memory_length, hidden_size].

        Example:
            .. code-block::

                import numpy as np
                import paddle
                from paddlenlp.transformers import ErnieDocForSequenceClassification
                from paddlenlp.transformers import ErnieDocTokenizer
                
                def get_related_pos(insts, seq_len, memory_len=128):
                    beg = seq_len + seq_len + memory_len
                    r_position = [list(range(beg - 1, seq_len - 1, -1)) + \
                                list(range(0, seq_len)) for i in range(len(insts))]
                    return np.array(r_position).astype('int64').reshape([len(insts), beg, 1])
                    
                tokenizer = ErnieDocTokenizer.from_pretrained('ernie-doc-base-zh')
                model = ErnieDocForSequenceClassification.from_pretrained('ernie-doc-base-zh', num_classes=2)

                inputs = tokenizer("欢迎使用百度飞桨！")
                inputs = {k:paddle.to_tensor([v + [0] * (128-len(v))]).unsqueeze(-1) for (k, v) in inputs.items()}
                
                memories = [paddle.zeros([1, 128, 768], dtype="float32") for _ in range(12)]
                position_ids = paddle.to_tensor(get_related_pos(inputs['input_ids'], 128, 128))
                attn_mask = paddle.ones([1, 128, 1])

                inputs['memories'] = memories
                inputs['position_ids'] = position_ids
                inputs['attn_mask'] = attn_mask

                outputs = model(**inputs)

                logits = outputs[0]
                mem = outputs[1]

        """
        _, pooled_output, mem = self.ernie_doc(
            input_ids, memories, token_type_ids, position_ids, attn_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits, mem


class ErnieDocForTokenClassification(ErnieDocPretrainedModel):
    """
    ErnieDoc Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        ernie_doc (:class:`ErnieDocModel`):
            An instance of :class:`ErnieDocModel`.
        num_classes (int):
            The number of classes.
        dropout (float, optional)
            The dropout ratio of last output. Default to 0.1.
    """

    def __init__(self, ernie_doc, num_classes, dropout=0.1):
        super(ErnieDocForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie_doc = ernie_doc  # allow ernie_doc to be config
        self.dropout = nn.Dropout(dropout, mode="upscale_in_train")
        self.linear = nn.Linear(self.ernie_doc.config["hidden_size"],
                                num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, memories, token_type_ids, position_ids,
                attn_mask):
        r"""
        The ErnieDocForTokenClassification forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                See :class:`ErnieDocModel`.
            memories (List[Tensor]):
                See :class:`ErnieDocModel`.
            token_type_ids (Tensor):
                See :class:`ErnieDocModel`.
                Defaults to None, which means no segment embeddings is added to token embeddings.
            position_ids (Tensor):
                See :class:`ErnieDocModel`.
            attn_mask (Tensor):
                See :class:`ErnieDocModel`.

        Returns:
            tuple : Returns tuple (`logits`, `mem`).

            With the fields:

            - `logits` (Tensor):
                A tensor containing the hidden-states of the model at the output of last layer.
                Each Tensor has a data type of `float32` and has a shape of [batch_size, sequence_length, num_classes].

            - `mem` (List[Tensor]):
                A list of pre-computed hidden-states. The length of the list is `n_layers`.
                Each element in the list is a Tensor with dtype `float32` and has a shape of
                [batch_size, memory_length, hidden_size].

        Example:
            .. code-block::

                import numpy as np
                import paddle
                from paddlenlp.transformers import ErnieDocForTokenClassification
                from paddlenlp.transformers import ErnieDocTokenizer
                
                def get_related_pos(insts, seq_len, memory_len=128):
                    beg = seq_len + seq_len + memory_len
                    r_position = [list(range(beg - 1, seq_len - 1, -1)) + \
                                list(range(0, seq_len)) for i in range(len(insts))]
                    return np.array(r_position).astype('int64').reshape([len(insts), beg, 1])
                    
                tokenizer = ErnieDocTokenizer.from_pretrained('ernie-doc-base-zh')
                model = ErnieDocForTokenClassification.from_pretrained('ernie-doc-base-zh', num_classes=2)

                inputs = tokenizer("欢迎使用百度飞桨！")
                inputs = {k:paddle.to_tensor([v + [0] * (128-len(v))]).unsqueeze(-1) for (k, v) in inputs.items()}
                
                memories = [paddle.zeros([1, 128, 768], dtype="float32") for _ in range(12)]
                position_ids = paddle.to_tensor(get_related_pos(inputs['input_ids'], 128, 128))
                attn_mask = paddle.ones([1, 128, 1])

                inputs['memories'] = memories
                inputs['position_ids'] = position_ids
                inputs['attn_mask'] = attn_mask

                outputs = model(**inputs)

                logits = outputs[0]
                mem = outputs[1]

        """
        sequence_output, _, mem = self.ernie_doc(
            input_ids, memories, token_type_ids, position_ids, attn_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)
        return logits, mem


class ErnieDocForQuestionAnswering(ErnieDocPretrainedModel):
    """
    ErnieDoc Model with a linear layer on top of the hidden-states
    output to compute `span_start_logits` and `span_end_logits`,
    designed for question-answering tasks like SQuAD.
    
    Args:
        ernie_doc (:class:`ErnieDocModel`):
            An instance of :class:`ErnieDocModel`.
        dropout (float, optional)
            The dropout ratio of last output. Default to 0.1.
    """

    def __init__(self, ernie_doc, dropout=0.1):
        super(ErnieDocForQuestionAnswering, self).__init__()
        self.ernie_doc = ernie_doc  # allow ernie_doc to be config
        self.dropout = nn.Dropout(dropout, mode="upscale_in_train")
        self.linear = nn.Linear(self.ernie_doc.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, memories, token_type_ids, position_ids,
                attn_mask):
        r"""
        The ErnieDocForQuestionAnswering forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                See :class:`ErnieDocModel`.
            memories (List[Tensor]):
                See :class:`ErnieDocModel`.
            token_type_ids (Tensor):
                See :class:`ErnieDocModel`.
            position_ids (Tensor):
                See :class:`ErnieDocModel`.
            attn_mask (Tensor):
                See :class:`ErnieDocModel`.

        Returns:
            tuple : Returns tuple (`start_logits`, `end_logits`, `mem`).

            With the fields:

            - `start_logits` (Tensor):
                A tensor of the input token classification logits, indicates the start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `end_logits` (Tensor):
                A tensor of the input token classification logits, indicates the end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `mem` (List[Tensor]):
                A list of pre-computed hidden-states. The length of the list is `n_layers`.
                Each element in the list is a Tensor with dtype `float32` and has a shape of
                [batch_size, memory_length, hidden_size].

        Example:
            .. code-block::

                import numpy as np
                import paddle
                from paddlenlp.transformers import ErnieDocForQuestionAnswering
                from paddlenlp.transformers import ErnieDocTokenizer
                
                def get_related_pos(insts, seq_len, memory_len=128):
                    beg = seq_len + seq_len + memory_len
                    r_position = [list(range(beg - 1, seq_len - 1, -1)) + \
                                list(range(0, seq_len)) for i in range(len(insts))]
                    return np.array(r_position).astype('int64').reshape([len(insts), beg, 1])
                    
                tokenizer = ErnieDocTokenizer.from_pretrained('ernie-doc-base-zh')
                model = ErnieDocForQuestionAnswering.from_pretrained('ernie-doc-base-zh')

                inputs = tokenizer("欢迎使用百度飞桨！")
                inputs = {k:paddle.to_tensor([v + [0] * (128-len(v))]).unsqueeze(-1) for (k, v) in inputs.items()}
                
                memories = [paddle.zeros([1, 128, 768], dtype="float32") for _ in range(12)]
                position_ids = paddle.to_tensor(get_related_pos(inputs['input_ids'], 128, 128))
                attn_mask = paddle.ones([1, 128, 1])

                inputs['memories'] = memories
                inputs['position_ids'] = position_ids
                inputs['attn_mask'] = attn_mask

                outputs = model(**inputs)

                start_logits = outputs[0]
                end_logits = outputs[1]
                mem = outputs[2]

        """
        sequence_output, _, mem = self.ernie_doc(
            input_ids, memories, token_type_ids, position_ids, attn_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)
        start_logits, end_logits = paddle.transpose(logits, perm=[2, 0, 1])
        return start_logits, end_logits, mem
