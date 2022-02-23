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

import math

import paddle
from paddle import tensor
import paddle.nn.functional as F
from paddle.nn import MultiHeadAttention, TransformerEncoderLayer, TransformerEncoder
from paddle.fluid.data_feeder import convert_dtype

from paddlenlp.utils.log import logger
from paddlenlp.transformers import PPMiniLMForSequenceClassification
from paddlenlp.transformers import TinyBertForPretraining
from paddlenlp.transformers import BertForSequenceClassification

__all__ = ['to_distill', 'calc_minilm_loss', 'calc_multi_relation_loss']


def calc_multi_relation_loss(loss_fct,
                             s,
                             t,
                             attn_mask,
                             num_relation_heads=0,
                             alpha=0.0,
                             beta=0.0):
    """
    Calculates loss for multiple Q-Q, K-K and V-V relation. It supports
    head-head relation, sample-sample relation and origin token-token relation.
    The final loss value could be balanced by weight `alpha` and `beta`.

    Args:
        loss_fct (callable):
            Loss function for distillation. It only supports kl_div loss now.
        s (Tensor):
            Q, K, V of Student.
        t (Tensor):
            Q, K, V of teacher.
        attn_mask (Tensor):
            Attention mask for relation.
        num_relation_heads (int):
            The number of relation heads. 0 means `num_relation_heads` equals
            to origin head num.
            Defaults to 0.
        alpha (float):
            The weight for head-head relation.
            Defaults to 0.0.
        beta (float):
            The weight for sample-sample relation.
            Defaults to 0.0.

    Returns:
        Tensor: Weighted loss of token-token loss, head-head loss and
            sample-sample loss.

    """
    # Initialize head_num
    if num_relation_heads > 0 and num_relation_heads != s.shape[1]:
        # s'shape: [bs, seq_len, head_num, head_dim]
        s = tensor.transpose(x=s, perm=[0, 2, 1, 3])
        # s'shape: [bs, seq_len, num_relation_heads, head_dim_new]
        s = tensor.reshape(x=s, shape=[0, 0, num_relation_heads, -1])
        s1 = tensor.transpose(x=s, perm=[0, 2, 1, 3])
    if num_relation_heads > 0 and num_relation_heads != t.shape[1]:
        t = tensor.transpose(x=t, perm=[0, 2, 1, 3])
        t = tensor.reshape(x=t, shape=[0, 0, num_relation_heads, -1])
        t1 = tensor.transpose(x=t, perm=[0, 2, 1, 3])

    s_head_dim, t_head_dim = s.shape[3], t.shape[3]

    if alpha + beta == 1.0:
        loss_token_token = 0.0
    else:
        scaled_dot_product_s1 = tensor.matmul(
            x=s1, y=s1, transpose_y=True) / math.sqrt(s_head_dim)
        del s1
        scaled_dot_product_s1 += attn_mask
        scaled_dot_product_t1 = tensor.matmul(
            x=t1, y=t1, transpose_y=True) / math.sqrt(t_head_dim)
        del t1
        scaled_dot_product_t1 += attn_mask
        loss_token_token = loss_fct(
            F.log_softmax(scaled_dot_product_s1),
            F.softmax(scaled_dot_product_t1))

    if alpha == 0.0:
        loss_head_head = 0.0
    else:
        scaled_dot_product_s = tensor.matmul(
            x=s, y=s, transpose_y=True) / math.sqrt(s_head_dim)
        attn_mask_head_head = tensor.transpose(x=attn_mask, perm=[0, 3, 1, 2])

        scaled_dot_product_s += attn_mask_head_head
        scaled_dot_product_t = tensor.matmul(
            x=t, y=t, transpose_y=True) / math.sqrt(t_head_dim)
        scaled_dot_product_t += attn_mask_head_head
        loss_head_head = loss_fct(
            F.log_softmax(scaled_dot_product_s),
            F.softmax(scaled_dot_product_t))
    if beta == 0.0:
        loss_sample_sample = 0.0
    else:
        s2 = tensor.transpose(x=s, perm=[1, 2, 0, 3])
        scaled_dot_product_s2 = tensor.matmul(
            x=s2, y=s2, transpose_y=True) / math.sqrt(s_head_dim)

        del s, s2
        # Shape: [seq_len, 1, batch_size, 1]
        attn_mask_sample_sample = tensor.transpose(
            x=attn_mask, perm=[3, 1, 0, 2])

        # Shape: [seq_len, head_num, batch_size, batch_size]
        scaled_dot_product_s2 += attn_mask_sample_sample
        t2 = tensor.transpose(x=t, perm=[1, 2, 0, 3])
        scaled_dot_product_t2 = tensor.matmul(
            x=t2, y=t2, transpose_y=True) / math.sqrt(t_head_dim)

        del t, t2
        scaled_dot_product_t2 += attn_mask_sample_sample
        loss_sample_sample = loss_fct(
            F.log_softmax(scaled_dot_product_s2),
            F.softmax(scaled_dot_product_t2))

    return (
        1 - alpha - beta
    ) * loss_token_token + alpha * loss_head_head + beta * loss_sample_sample


def calc_minilm_loss(loss_fct, s, t, attn_mask, num_relation_heads=0):
    """
    Calculates loss for Q-Q, K-K, V-V relation from MiniLMv2.
    Args:
        loss_fct (callable):
            Loss function for distillation. It only supports kl_div loss now.
        s (Tensor):
            Q, K, V of Student.
        t (Tensor):
            Q, K, V of teacher.
        attn_mask (Tensor):
            Attention mask for relation.
        num_relation_heads (int):
            The number of relation heads. 0 means `num_relation_heads` equals
            to origin head num.
            Defaults to 0.

    Returns:
        Tensor: MiniLM loss value.

    """
    # Initialize head_num
    if num_relation_heads > 0 and num_relation_heads != s.shape[1]:
        # s'shape: [bs, seq_len, head_num, head_dim]
        s = tensor.transpose(x=s, perm=[0, 2, 1, 3])
        # s'shape: [bs, seq_len, num_relation_heads, head_dim_new]
        s = tensor.reshape(x=s, shape=[0, 0, num_relation_heads, -1])
        # s' shape: [bs, num_relation_heads, seq_len, head_dim_new]
        s = tensor.transpose(x=s, perm=[0, 2, 1, 3])
    if num_relation_heads > 0 and num_relation_heads != t.shape[1]:
        t = tensor.transpose(x=t, perm=[0, 2, 1, 3])
        t = tensor.reshape(x=t, shape=[0, 0, num_relation_heads, -1])
        t = tensor.transpose(x=t, perm=[0, 2, 1, 3])

    s_head_dim, t_head_dim = s.shape[3], t.shape[3]
    scaled_dot_product_s = tensor.matmul(
        x=s, y=s, transpose_y=True) / math.sqrt(s_head_dim)
    del s
    scaled_dot_product_s += attn_mask

    scaled_dot_product_t = tensor.matmul(
        x=t, y=t, transpose_y=True) / math.sqrt(t_head_dim)
    del t
    scaled_dot_product_t += attn_mask
    loss = loss_fct(
        F.log_softmax(scaled_dot_product_s), F.softmax(scaled_dot_product_t))
    return loss


def to_distill(self,
               return_qkv=False,
               return_attentions=False,
               return_layer_outputs=False,
               layer_index=-1):
    """
    Can be bound to object with transformer encoder layers, and make model
    expose attributes `outputs.q`, `outputs.k`, `outputs.v`,
    `outputs.scaled_qks`, `outputs.hidden_states`and `outputs.attentions` of
    the object for distillation.
    It could be returned intermediate tensor using in MiniLM and TinyBERT
    strategy.
    """
    logger.warning("`to_distill` is an experimental API and subject to change.")
    MultiHeadAttention._forward = attention_forward
    TransformerEncoderLayer._forward = transformer_encoder_layer_forward
    TransformerEncoder._forward = transformer_encoder_forward
    BertForSequenceClassification._forward = bert_forward

    if return_qkv:
        # forward function of student class should be replaced for distributed training.
        TinyBertForPretraining._forward = minilm_pretraining_forward
        PPMiniLMForSequenceClassification._forward = minilm_pretraining_forward
    else:
        TinyBertForPretraining._forward = tinybert_forward

    def init_func(layer):
        if isinstance(layer, (MultiHeadAttention, TransformerEncoderLayer,
                              TransformerEncoder, TinyBertForPretraining,
                              BertForSequenceClassification,
                              PPMiniLMForSequenceClassification)):
            layer.forward = layer._forward
            if isinstance(layer, TransformerEncoder):
                layer.return_layer_outputs = return_layer_outputs
                layer.layer_index = layer_index
            if isinstance(layer, MultiHeadAttention):
                layer.return_attentions = return_attentions
                layer.return_qkv = return_qkv

    for layer in self.children():
        layer.apply(init_func)

    base_model_prefix = self._layers.base_model_prefix if isinstance(
        self, paddle.DataParallel) else self.base_model_prefix

    # For distribute training
    if isinstance(self, paddle.DataParallel):
        if hasattr(self._layers, base_model_prefix):
            self.outputs = getattr(self._layers, base_model_prefix).encoder
        else:
            self.outputs = self._layers.encoder
    else:
        if hasattr(self, base_model_prefix):
            self.outputs = getattr(self, base_model_prefix).encoder
        else:
            self.outputs = self.encoder
    return self


def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


def attention_forward(self,
                      query,
                      key=None,
                      value=None,
                      attn_mask=None,
                      cache=None):
    """
    Redefines the `forward` function of `paddle.nn.MultiHeadAttention`.
    """
    key = query if key is None else key
    value = query if value is None else value
    # Computes q ,k ,v
    if cache is None:
        q, k, v = self._prepare_qkv(query, key, value, cache)
    else:
        q, k, v, cache = self._prepare_qkv(query, key, value, cache)

    # Scale dot product attention
    product = tensor.matmul(x=q, y=k, transpose_y=True)
    product /= math.sqrt(self.head_dim)

    if attn_mask is not None:
        # Support bool or int mask
        attn_mask = _convert_attention_mask(attn_mask, product.dtype)
        product = product + attn_mask

    self.attention_matrix = product if self.return_attentions else None
    weights = F.softmax(product)
    if self.dropout:
        weights = F.dropout(
            weights,
            self.dropout,
            training=self.training,
            mode="upscale_in_train")

    out = tensor.matmul(weights, v)
    if self.return_qkv:
        self.q = q
        self.k = k
        self.v = v

    # Combine heads
    out = tensor.transpose(out, perm=[0, 2, 1, 3])
    out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

    # Project to output
    out = self.out_proj(out)

    outs = [out]
    if self.need_weights:
        outs.append(weights)
    if cache is not None:
        outs.append(cache)
    return out if len(outs) == 1 else tuple(outs)


def transformer_encoder_layer_forward(self, src, src_mask=None, cache=None):
    """
    Redefines the `forward` function of `paddle.nn.TransformerEncoderLayer`.
    """
    src_mask = _convert_attention_mask(src_mask, src.dtype)

    residual = src
    if self.normalize_before:
        src = self.norm1(src)
    # Add cache for encoder for the usage like UniLM
    if cache is None:
        src = self.self_attn(src, src, src, src_mask)
    else:
        src, incremental_cache = self.self_attn(src, src, src, src_mask, cache)
    src = residual + self.dropout1(src)
    if not self.normalize_before:
        src = self.norm1(src)

    residual = src
    if self.normalize_before:
        src = self.norm2(src)
    src = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = residual + self.dropout2(src)
    if not self.normalize_before:
        src = self.norm2(src)
    if hasattr(self.self_attn, 'attention_matrix'):
        self.attention_matrix = self.self_attn.attention_matrix
    if hasattr(self.self_attn, 'q'):
        self.q = self.self_attn.q
        self.k = self.self_attn.k
        self.v = self.self_attn.v
    return src if cache is None else (src, incremental_cache)


def transformer_encoder_forward(self, src, src_mask=None, cache=None):
    """
    Redefines the `forward` function of `paddle.nn.TransformerEncoder`.
    """
    src_mask = _convert_attention_mask(src_mask, src.dtype)

    output = src
    new_caches = []

    self.attentions = []
    self.hidden_states = []

    for i, mod in enumerate(self.layers):
        if self.return_layer_outputs:
            self.hidden_states.append(output)
        if cache is None:
            output = mod(output, src_mask=src_mask)
        else:
            output, new_cache = mod(output, src_mask=src_mask, cache=cache[i])
            new_caches.append(new_cache)
        if hasattr(mod, 'attention_matrix'):
            self.attentions.append(mod.attention_matrix)
        if i == self.layer_index and hasattr(mod, 'q'):
            self.q = mod.q
            self.k = mod.k
            self.v = mod.v

    if self.norm is not None:
        output = self.norm(output)
    if self.return_layer_outputs:
        self.hidden_states.append(output)
    return output if cache is None else (output, new_caches)


def minilm_pretraining_forward(self,
                               input_ids,
                               token_type_ids=None,
                               attention_mask=None):
    """
    Replaces `forward` function while using multi gpus to train. If training on
    single GPU, this `forward` could not be replaced.
    The type of `self` should inherit from base class of pretrained LMs, such as
    `TinyBertForPretraining`.
    Strategy MINILM only needs q, k and v of transformers.
    """
    assert hasattr(self, self.base_model_prefix), \
        "Student class should inherit from %s" % (self.base_model_class)
    model = getattr(self, self.base_model_prefix)
    encoder = model.encoder

    sequence_output, pooled_output = model(input_ids, token_type_ids,
                                           attention_mask)
    return encoder.q, encoder.k, encoder.v


def tinybert_forward(self, input_ids, token_type_ids=None, attention_mask=None):
    """
    Replaces `forward` function while using multi gpus to train.
    """
    assert hasattr(self, self.base_model_prefix), \
        "Student class should inherit from %s" % (self.base_model_class)
    model = getattr(self, self.base_model_prefix)
    encoder = model.encoder

    sequence_output, pooled_output = model(input_ids, token_type_ids,
                                           attention_mask)
    for i in range(len(encoder.hidden_states)):
        # While using tinybert-4l-312d, tinybert-6l-768d, tinybert-4l-312d-zh,
        # tinybert-6l-768d-zh
        # While using tinybert-4l-312d-v2, tinybert-6l-768d-v2
        # encoder.hidden_states[i] = self.tinybert.fit_dense(encoder.hidden_states[i])
        encoder.hidden_states[i] = self.tinybert.fit_denses[i](
            encoder.hidden_states[i])

    return encoder.attentions, encoder.hidden_states


def bert_forward(self, input_ids, token_type_ids=None, attention_mask=None):
    """
    Replaces `forward` function while using multi gpus to train.
    """
    assert hasattr(self, self.base_model_prefix), \
        "Student class should inherit from %s" % (self.base_model_class)
    model = getattr(self, self.base_model_prefix)
    encoder = model.encoder

    sequence_output, pooled_output = model(input_ids, token_type_ids,
                                           attention_mask)
    return encoder.attentions, encoder.hidden_states