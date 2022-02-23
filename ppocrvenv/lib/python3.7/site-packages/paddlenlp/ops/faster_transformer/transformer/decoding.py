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
import numpy as np
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid.framework import in_dygraph_mode

from paddle.fluid.layer_helper import LayerHelper
import paddle

from paddlenlp.ops.ext_utils import load, LOADED_EXT
from paddlenlp.utils.log import logger


def infer_transformer_decoding(
        enc_output, memory_seq_lens, word_emb, slf_ln_weight, slf_ln_bias,
        slf_q_weight, slf_q_bias, slf_k_weight, slf_k_bias, slf_v_weight,
        slf_v_bias, slf_out_weight, slf_out_bias, cross_ln_weight,
        cross_ln_bias, cross_q_weight, cross_q_bias, cross_k_weight,
        cross_k_bias, cross_v_weight, cross_v_bias, cross_out_weight,
        cross_out_bias, ffn_ln_weight, ffn_ln_bias, ffn_inter_weight,
        ffn_inter_bias, ffn_out_weight, ffn_out_bias, decoder_ln_weight,
        decoder_ln_bias, linear_weight, linear_bias, pos_emb,
        _decoding_strategy, _beam_size, _topk, _topp, _n_head, _size_per_head,
        _n_layer, _bos_id, _eos_id, _max_out_len, _diversity_rate, _rel_len,
        _alpha):
    helper = LayerHelper('fusion_decoding', **locals())

    inputs = {
        'Input': enc_output,
        'MemSeqLen': memory_seq_lens,
        'WordEmbedding': word_emb,
        'SelfLayernormWeight@VECTOR': slf_ln_weight,
        'SelfLayernormBias@VECTOR': slf_ln_bias,
        'SelfQueryWeight@VECTOR': slf_q_weight,
        'SelfQueryBias@VECTOR': slf_q_bias,
        'SelfKeyWeight@VECTOR': slf_k_weight,
        'SelfKeyBias@VECTOR': slf_k_bias,
        'SelfValueWeight@VECTOR': slf_v_weight,
        'SelfValueBias@VECTOR': slf_v_bias,
        'SelfOutWeight@VECTOR': slf_out_weight,
        'SelfOutBias@VECTOR': slf_out_bias,
        'CrossLayernormWeight@VECTOR': cross_ln_weight,
        'CrossLayernormBias@VECTOR': cross_ln_bias,
        'CrossQueryWeight@VECTOR': cross_q_weight,
        'CrossQueryBias@VECTOR': cross_q_bias,
        'CrossKeyWeight@VECTOR': cross_k_weight,
        'CrossKeyBias@VECTOR': cross_k_bias,
        'CrossValueWeight@VECTOR': cross_v_weight,
        'CrossValueBias@VECTOR': cross_v_bias,
        'CrossOutWeight@VECTOR': cross_out_weight,
        'CrossOutBias@VECTOR': cross_out_bias,
        'FFNLayernormWeight@VECTOR': ffn_ln_weight,
        'FFNLayernormBias@VECTOR': ffn_ln_bias,
        'FFNInterWeight@VECTOR': ffn_inter_weight,
        'FFNInterBias@VECTOR': ffn_inter_bias,
        'FFNOutWeight@VECTOR': ffn_out_weight,
        'FFNOutBias@VECTOR': ffn_out_bias,
        'DecoderLayernormWeight': decoder_ln_weight,
        'DecoderLayernormBias': decoder_ln_bias,
        'EmbWeight': linear_weight,
        'EmbBias': linear_bias,
        'PositionEncEmb': pos_emb
    }

    attrs = {
        'decoding_strategy': _decoding_strategy,
        'beam_size': _beam_size,
        'topk': _topk,
        'topp': _topp,
        'n_head': _n_head,
        'size_per_head': _size_per_head,
        'num_layer': _n_layer,
        'bos_id': _bos_id,
        'eos_id': _eos_id,
        'max_len': _max_out_len,
        'beam_search_diversity_rate': _diversity_rate,
        "rel_len": _rel_len,
        "alpha": _alpha
    }

    output_ids = helper.create_variable(dtype="int32")
    parent_ids = helper.create_variable(dtype="int32")
    sequence_length = helper.create_variable(dtype="int32")

    outputs = {
        'OutputIds': output_ids,
        'ParentIds': parent_ids,
        'SequenceLength': sequence_length
    }

    helper.append_op(
        type='fusion_decoding', inputs=inputs, outputs=outputs, attrs=attrs)

    return output_ids, parent_ids, sequence_length


def infer_force_decoding(
        enc_output, memory_seq_lens, word_emb, slf_ln_weight, slf_ln_bias,
        slf_q_weight, slf_q_bias, slf_k_weight, slf_k_bias, slf_v_weight,
        slf_v_bias, slf_out_weight, slf_out_bias, cross_ln_weight,
        cross_ln_bias, cross_q_weight, cross_q_bias, cross_k_weight,
        cross_k_bias, cross_v_weight, cross_v_bias, cross_out_weight,
        cross_out_bias, ffn_ln_weight, ffn_ln_bias, ffn_inter_weight,
        ffn_inter_bias, ffn_out_weight, ffn_out_bias, decoder_ln_weight,
        decoder_ln_bias, linear_weight, linear_bias, pos_emb, trg_word,
        _decoding_strategy, _beam_size, _topk, _topp, _n_head, _size_per_head,
        _n_layer, _bos_id, _eos_id, _max_out_len, _diversity_rate, _rel_len,
        _alpha):
    helper = LayerHelper('fusion_force_decoding', **locals())

    inputs = {
        'Input': enc_output,
        'MemSeqLen': memory_seq_lens,
        'WordEmbedding': word_emb,
        'SelfLayernormWeight@VECTOR': slf_ln_weight,
        'SelfLayernormBias@VECTOR': slf_ln_bias,
        'SelfQueryWeight@VECTOR': slf_q_weight,
        'SelfQueryBias@VECTOR': slf_q_bias,
        'SelfKeyWeight@VECTOR': slf_k_weight,
        'SelfKeyBias@VECTOR': slf_k_bias,
        'SelfValueWeight@VECTOR': slf_v_weight,
        'SelfValueBias@VECTOR': slf_v_bias,
        'SelfOutWeight@VECTOR': slf_out_weight,
        'SelfOutBias@VECTOR': slf_out_bias,
        'CrossLayernormWeight@VECTOR': cross_ln_weight,
        'CrossLayernormBias@VECTOR': cross_ln_bias,
        'CrossQueryWeight@VECTOR': cross_q_weight,
        'CrossQueryBias@VECTOR': cross_q_bias,
        'CrossKeyWeight@VECTOR': cross_k_weight,
        'CrossKeyBias@VECTOR': cross_k_bias,
        'CrossValueWeight@VECTOR': cross_v_weight,
        'CrossValueBias@VECTOR': cross_v_bias,
        'CrossOutWeight@VECTOR': cross_out_weight,
        'CrossOutBias@VECTOR': cross_out_bias,
        'FFNLayernormWeight@VECTOR': ffn_ln_weight,
        'FFNLayernormBias@VECTOR': ffn_ln_bias,
        'FFNInterWeight@VECTOR': ffn_inter_weight,
        'FFNInterBias@VECTOR': ffn_inter_bias,
        'FFNOutWeight@VECTOR': ffn_out_weight,
        'FFNOutBias@VECTOR': ffn_out_bias,
        'DecoderLayernormWeight': decoder_ln_weight,
        'DecoderLayernormBias': decoder_ln_bias,
        'EmbWeight': linear_weight,
        'EmbBias': linear_bias,
        'PositionEncEmb': pos_emb,
        # The input of custom op must be given.
        # Dispensable() and Intermediate() are not supported.
        'TrgWord': trg_word
    }

    attrs = {
        'decoding_strategy': _decoding_strategy,
        'beam_size': _beam_size,
        'topk': _topk,
        'topp': _topp,
        'n_head': _n_head,
        'size_per_head': _size_per_head,
        'num_layer': _n_layer,
        'bos_id': _bos_id,
        'eos_id': _eos_id,
        'max_len': _max_out_len,
        'beam_search_diversity_rate': _diversity_rate,
        "rel_len": _rel_len,
        "alpha": _alpha
    }

    output_ids = helper.create_variable(dtype="int32")
    parent_ids = helper.create_variable(dtype="int32")
    sequence_length = helper.create_variable(dtype="int32")

    outputs = {
        'OutputIds': output_ids,
        'ParentIds': parent_ids,
        'SequenceLength': sequence_length
    }

    helper.append_op(
        type='fusion_force_decoding',
        inputs=inputs,
        outputs=outputs,
        attrs=attrs)

    return output_ids, parent_ids, sequence_length


def infer_gpt_decoding(
        input, attn_mask, mem_seq_len, word_emb, slf_ln_weight, slf_ln_bias,
        slf_q_weight, slf_q_bias, slf_k_weight, slf_k_bias, slf_v_weight,
        slf_v_bias, slf_out_weight, slf_out_bias, ffn_ln_weight, ffn_ln_bias,
        ffn_inter_weight, ffn_inter_bias, ffn_out_weight, ffn_out_bias,
        decoder_ln_weight, decoder_ln_bias, pos_emb, linear_weight, topk, topp,
        max_out_len, head_num, size_per_head, num_layer, bos_id, eos_id,
        temperature, use_fp16_decoding):
    helper = LayerHelper('fusion_gpt', **locals())

    inputs = {
        "Input": input,
        "AttentionMask": attn_mask,
        "StartLength": mem_seq_len,
        "WordEmbedding": word_emb,
        "SelfLayernormWeight@VECTOR": slf_ln_weight,
        "SelfLayernormBias@VECTOR": slf_ln_bias,
        "SelfQueryWeight@VECTOR": slf_q_weight,
        "SelfQueryBias@VECTOR": slf_q_bias,
        "SelfKeyWeight@VECTOR": slf_k_weight,
        "SelfKeyBias@VECTOR": slf_k_bias,
        "SelfValueWeight@VECTOR": slf_v_weight,
        "SelfValueBias@VECTOR": slf_v_bias,
        "SelfOutWeight@VECTOR": slf_out_weight,
        "SelfOutBias@VECTOR": slf_out_bias,
        "FFNLayernormWeight@VECTOR": ffn_ln_weight,
        "FFNLayernormBias@VECTOR": ffn_ln_bias,
        "FFNInterWeight@VECTOR": ffn_inter_weight,
        "FFNInterBias@VECTOR": ffn_inter_bias,
        "FFNOutWeight@VECTOR": ffn_out_weight,
        "FFNOutBias@VECTOR": ffn_out_bias,
        "DecoderLayernormWeight": decoder_ln_weight,
        "DecoderLayernormBias": decoder_ln_bias,
        "PositionEncEmb": pos_emb,
        "EmbWeight": linear_weight
    }

    attrs = {
        "topk": topk,
        "topp": topp,
        "max_len": max_out_len,
        "n_head": head_num,
        "size_per_head": size_per_head,
        "num_layer": num_layer,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "temperature": temperature,
        "use_fp16": use_fp16_decoding
    }

    output_ids = helper.create_variable(dtype="int32")
    outputs = {'OutputIds': output_ids}

    helper.append_op(
        type='fusion_gpt', inputs=inputs, outputs=outputs, attrs=attrs)

    return output_ids


def infer_unified_decoding(
        cache_k, cache_v, memory_seq_lens, type_id, logits_mask, word_emb,
        slf_ln_weight, slf_ln_bias, slf_q_weight, slf_q_bias, slf_k_weight,
        slf_k_bias, slf_v_weight, slf_v_bias, slf_out_weight, slf_out_bias,
        ffn_ln_weight, ffn_ln_bias, ffn_inter_weight, ffn_inter_bias,
        ffn_out_weight, ffn_out_bias, decoder_ln_weight, decoder_ln_bias,
        trans_weight, trans_bias, lm_ln_weight, lm_ln_bias, linear_weight,
        linear_bias, pos_emb, type_emb, _decoding_strategy, _beam_size, _topk,
        _topp, _n_head, _size_per_head, _n_layer, _bos_id, _eos_id,
        _max_out_len, _diversity_rate, _unk_id, _mask_id, _temperature,
        _len_penalty, _normalize_before, _pos_bias, _hidden_act, _rel_len,
        _early_stopping):
    helper = LayerHelper('fusion_unified_decoding', **locals())

    inputs = {
        "CacheK@VECTOR": cache_k,
        "CacheV@VECTOR": cache_v,
        "MemSeqLen": memory_seq_lens,
        "TypeId": type_id,
        "LogitsMask": logits_mask,
        "WordEmbedding": word_emb,
        "SelfLayernormWeight@VECTOR": slf_ln_weight,
        "SelfLayernormBias@VECTOR": slf_ln_bias,
        "SelfQueryWeight@VECTOR": slf_q_weight,
        "SelfQueryBias@VECTOR": slf_q_bias,
        "SelfKeyWeight@VECTOR": slf_k_weight,
        "SelfKeyBias@VECTOR": slf_k_bias,
        "SelfValueWeight@VECTOR": slf_v_weight,
        "SelfValueBias@VECTOR": slf_v_bias,
        "SelfOutWeight@VECTOR": slf_out_weight,
        "SelfOutBias@VECTOR": slf_out_bias,
        "FFNLayernormWeight@VECTOR": ffn_ln_weight,
        "FFNLayernormBias@VECTOR": ffn_ln_bias,
        "FFNInterWeight@VECTOR": ffn_inter_weight,
        "FFNInterBias@VECTOR": ffn_inter_bias,
        "FFNOutWeight@VECTOR": ffn_out_weight,
        "FFNOutBias@VECTOR": ffn_out_bias,
        "DecoderLayernormWeight": decoder_ln_weight,
        "DecoderLayernormBias": decoder_ln_bias,
        "TransWeight": trans_weight,
        "TransBias": trans_bias,
        "LMLayernormWeight": lm_ln_weight,
        "LMLayernormBias": lm_ln_bias,
        "EmbWeight": linear_weight,
        "EmbBias": linear_bias,
        "PositionEncEmb": pos_emb,
        "TypeEmb": type_emb
    }

    attrs = {
        "decoding_strategy": _decoding_strategy,
        "beam_size": _beam_size,
        "topk": _topk,
        "topp": _topp,
        "n_head": _n_head,
        "size_per_head": _size_per_head,
        "num_layer": _n_layer,
        "bos_id": _bos_id,
        "eos_id": _eos_id,
        "max_len": _max_out_len,
        "beam_search_diversity_rate": _diversity_rate,
        "unk_id": _unk_id,
        "mask_id": _mask_id,
        "temperature": _temperature,
        "len_penalty": _len_penalty,
        "normalize_before": _normalize_before,
        "pos_bias": _pos_bias,
        "hidden_act": _hidden_act,
        "rel_len": _rel_len,
        "early_stopping": _early_stopping
    }

    output_ids = helper.create_variable(dtype="int32")
    parent_ids = helper.create_variable(dtype="int32")
    sequence_length = helper.create_variable(dtype="int32")

    outputs = {
        'OutputIds': output_ids,
        'ParentIds': parent_ids,
        'SequenceLength': sequence_length
    }

    helper.append_op(
        type='fusion_unified_decoding',
        inputs=inputs,
        outputs=outputs,
        attrs=attrs)

    return output_ids, parent_ids, sequence_length


def infer_bart_decoding(
        enc_output, memory_seq_lens, word_emb, slf_ln_weight, slf_ln_bias,
        slf_q_weight, slf_q_bias, slf_k_weight, slf_k_bias, slf_v_weight,
        slf_v_bias, slf_out_weight, slf_out_bias, cross_ln_weight,
        cross_ln_bias, cross_q_weight, cross_q_bias, cross_k_weight,
        cross_k_bias, cross_v_weight, cross_v_bias, cross_out_weight,
        cross_out_bias, ffn_ln_weight, ffn_ln_bias, ffn_inter_weight,
        ffn_inter_bias, ffn_out_weight, ffn_out_bias, decoder_ln_weight,
        decoder_ln_bias, linear_weight, linear_bias, pos_emb,
        _decoding_strategy, _beam_size, _topk, _topp, _n_head, _size_per_head,
        _n_layer, _bos_id, _eos_id, _max_out_len, _diversity_rate, _rel_len,
        _alpha, _early_stopping):

    helper = LayerHelper('fusion_bart_decoding', **locals())

    inputs = {
        'Input': enc_output,
        'MemSeqLen': memory_seq_lens,
        'WordEmbedding': word_emb,
        'SelfLayernormWeight@VECTOR': slf_ln_weight,
        'SelfLayernormBias@VECTOR': slf_ln_bias,
        'SelfQueryWeight@VECTOR': slf_q_weight,
        'SelfQueryBias@VECTOR': slf_q_bias,
        'SelfKeyWeight@VECTOR': slf_k_weight,
        'SelfKeyBias@VECTOR': slf_k_bias,
        'SelfValueWeight@VECTOR': slf_v_weight,
        'SelfValueBias@VECTOR': slf_v_bias,
        'SelfOutWeight@VECTOR': slf_out_weight,
        'SelfOutBias@VECTOR': slf_out_bias,
        'CrossLayernormWeight@VECTOR': cross_ln_weight,
        'CrossLayernormBias@VECTOR': cross_ln_bias,
        'CrossQueryWeight@VECTOR': cross_q_weight,
        'CrossQueryBias@VECTOR': cross_q_bias,
        'CrossKeyWeight@VECTOR': cross_k_weight,
        'CrossKeyBias@VECTOR': cross_k_bias,
        'CrossValueWeight@VECTOR': cross_v_weight,
        'CrossValueBias@VECTOR': cross_v_bias,
        'CrossOutWeight@VECTOR': cross_out_weight,
        'CrossOutBias@VECTOR': cross_out_bias,
        'FFNLayernormWeight@VECTOR': ffn_ln_weight,
        'FFNLayernormBias@VECTOR': ffn_ln_bias,
        'FFNInterWeight@VECTOR': ffn_inter_weight,
        'FFNInterBias@VECTOR': ffn_inter_bias,
        'FFNOutWeight@VECTOR': ffn_out_weight,
        'FFNOutBias@VECTOR': ffn_out_bias,
        'DecoderLayernormWeight': decoder_ln_weight,
        'DecoderLayernormBias': decoder_ln_bias,
        'EmbWeight': linear_weight,
        'EmbBias': linear_bias,
        'PositionEncEmb': pos_emb
    }

    attrs = {
        'decoding_strategy': _decoding_strategy,
        'beam_size': _beam_size,
        'topk': _topk,
        'topp': _topp,
        'n_head': _n_head,
        'size_per_head': _size_per_head,
        'num_layer': _n_layer,
        'bos_id': _bos_id,
        'eos_id': _eos_id,
        'max_len': _max_out_len,
        'beam_search_diversity_rate': _diversity_rate,
        "rel_len": _rel_len,
        "alpha": _alpha,
        "early_stopping": _early_stopping
    }

    output_ids = helper.create_variable(dtype="int32")
    parent_ids = helper.create_variable(dtype="int32")
    sequence_length = helper.create_variable(dtype="int32")

    outputs = {
        'OutputIds': output_ids,
        'ParentIds': parent_ids,
        'SequenceLength': sequence_length
    }

    helper.append_op(
        type='fusion_bart_decoding',
        inputs=inputs,
        outputs=outputs,
        attrs=attrs)

    return output_ids, parent_ids, sequence_length


def infer_mbart_decoding(
        enc_output, memory_seq_lens, word_emb, slf_ln_weight, slf_ln_bias,
        slf_q_weight, slf_q_bias, slf_k_weight, slf_k_bias, slf_v_weight,
        slf_v_bias, slf_out_weight, slf_out_bias, cross_ln_weight,
        cross_ln_bias, cross_q_weight, cross_q_bias, cross_k_weight,
        cross_k_bias, cross_v_weight, cross_v_bias, cross_out_weight,
        cross_out_bias, ffn_ln_weight, ffn_ln_bias, ffn_inter_weight,
        ffn_inter_bias, ffn_out_weight, ffn_out_bias, decoder_ln_weight,
        decoder_ln_bias, mbart_ln_weight, mbart_ln_bias, linear_weight,
        linear_bias, pos_emb, trg_word, _decoding_strategy, _beam_size, _topk,
        _topp, _n_head, _size_per_head, _n_layer, _bos_id, _eos_id,
        _max_out_len, _diversity_rate, _rel_len, _alpha, _temperature,
        _early_stopping, _hidden_act):
    helper = LayerHelper('fusion_mbart_decoding', **locals())

    inputs = {
        'Input': enc_output,
        'MemSeqLen': memory_seq_lens,
        'WordEmbedding': word_emb,
        'SelfLayernormWeight@VECTOR': slf_ln_weight,
        'SelfLayernormBias@VECTOR': slf_ln_bias,
        'SelfQueryWeight@VECTOR': slf_q_weight,
        'SelfQueryBias@VECTOR': slf_q_bias,
        'SelfKeyWeight@VECTOR': slf_k_weight,
        'SelfKeyBias@VECTOR': slf_k_bias,
        'SelfValueWeight@VECTOR': slf_v_weight,
        'SelfValueBias@VECTOR': slf_v_bias,
        'SelfOutWeight@VECTOR': slf_out_weight,
        'SelfOutBias@VECTOR': slf_out_bias,
        'CrossLayernormWeight@VECTOR': cross_ln_weight,
        'CrossLayernormBias@VECTOR': cross_ln_bias,
        'CrossQueryWeight@VECTOR': cross_q_weight,
        'CrossQueryBias@VECTOR': cross_q_bias,
        'CrossKeyWeight@VECTOR': cross_k_weight,
        'CrossKeyBias@VECTOR': cross_k_bias,
        'CrossValueWeight@VECTOR': cross_v_weight,
        'CrossValueBias@VECTOR': cross_v_bias,
        'CrossOutWeight@VECTOR': cross_out_weight,
        'CrossOutBias@VECTOR': cross_out_bias,
        'FFNLayernormWeight@VECTOR': ffn_ln_weight,
        'FFNLayernormBias@VECTOR': ffn_ln_bias,
        'FFNInterWeight@VECTOR': ffn_inter_weight,
        'FFNInterBias@VECTOR': ffn_inter_bias,
        'FFNOutWeight@VECTOR': ffn_out_weight,
        'FFNOutBias@VECTOR': ffn_out_bias,
        'DecoderLayernormWeight': decoder_ln_weight,
        'DecoderLayernormBias': decoder_ln_bias,
        'MBARTLayernormWeight': mbart_ln_weight,
        'MBARTLayernormBias': mbart_ln_bias,
        'EmbWeight': linear_weight,
        'EmbBias': linear_bias,
        'PositionEncEmb': pos_emb,
        # The input of custom op must be given.
        # Dispensable() and Intermediate() are not supported. 
        'TrgWord': trg_word
    }

    attrs = {
        'decoding_strategy': _decoding_strategy,
        'beam_size': _beam_size,
        'topk': _topk,
        'topp': _topp,
        'n_head': _n_head,
        'size_per_head': _size_per_head,
        'num_layer': _n_layer,
        'bos_id': _bos_id,
        'eos_id': _eos_id,
        'max_len': _max_out_len,
        'beam_search_diversity_rate': _diversity_rate,
        "rel_len": _rel_len,
        "alpha": _alpha,
        "temperature": _temperature,
        "early_stopping": _early_stopping,
        "hidden_act": _hidden_act
    }

    output_ids = helper.create_variable(dtype="int32")
    parent_ids = helper.create_variable(dtype="int32")
    sequence_length = helper.create_variable(dtype="int32")

    outputs = {
        'OutputIds': output_ids,
        'ParentIds': parent_ids,
        'SequenceLength': sequence_length
    }

    helper.append_op(
        type='fusion_mbart_decoding',
        inputs=inputs,
        outputs=outputs,
        attrs=attrs)

    return output_ids, parent_ids, sequence_length


def finalize(beam_size,
             output_ids,
             parent_ids,
             out_seq_lens,
             forced_eos_token_id=None,
             max_seq_len=None,
             decoding_strategy="beam_search"):
    if max_seq_len is None:
        max_seq_len = paddle.max(out_seq_lens)
    ids = paddle.slice(output_ids, [0], [0], [max_seq_len])
    if decoding_strategy.startswith("beam_search"):
        parent_ids = paddle.slice(parent_ids, [0], [0], [max_seq_len]) % (
            beam_size * 2 if decoding_strategy.endswith("_v2") or
            decoding_strategy.endswith("_v3") else beam_size)
        ids = paddle.nn.functional.gather_tree(ids, parent_ids)
        if forced_eos_token_id is not None:
            ids[-1, :, :] = forced_eos_token_id
    else:
        if forced_eos_token_id is not None:
            ids[-1, :] = forced_eos_token_id
    return ids


def transfer_param(p, is_bias=False, restore_data=False):
    param_shape = p.shape
    # Maybe we need allow users using `model.to('float16')` to use fp16 by this.
    if (p.dtype == paddle.float16): return p
    if restore_data:
        if in_dygraph_mode():
            param_data = p.numpy()
            # Creating parameters with Assign initializer is too slow. Maybe we
            # can cast to fp16 directly and get a tensor, while we do it more
            # elaborately to get a ParamBase. Also note `VarBase.set_value`
            # enforce the same dtype and can not be used directly.
            new_p = type(p)(shape=param_shape, dtype="float16", is_bias=is_bias)
            new_p.value().get_tensor().set(
                param_data.astype("float16"),
                paddle.fluid.framework._current_expected_place())
            return new_p
        else:
            param_data = np.array(paddle.static.global_scope().find_var(p.name)
                                  .get_tensor())
    return paddle.create_parameter(
        shape=param_shape,
        dtype="float16",
        is_bias=is_bias,
        default_initializer=paddle.nn.initializer.Assign(param_data)
        if restore_data else None)


class InferTransformerDecoding(nn.Layer):
    def __init__(self,
                 decoder,
                 word_embedding,
                 positional_embedding,
                 linear,
                 num_decoder_layers,
                 n_head,
                 d_model,
                 bos_id=0,
                 eos_id=1,
                 decoding_strategy="beam_search",
                 beam_size=4,
                 topk=1,
                 topp=0.0,
                 max_out_len=256,
                 diversity_rate=0.0,
                 decoding_lib=None,
                 use_fp16_decoding=False,
                 rel_len=False,
                 alpha=0.6):
        # if decoding_lib is None:
        #     raise ValueError(
        #         "The args decoding_lib must be set to use FasterTransformer. ")
        # elif not os.path.exists(decoding_lib):
        #     raise ValueError("The path to decoding lib is not exist.")
        if decoding_lib is not None and os.path.isfile(decoding_lib):
            # Maybe it has been loadad by `ext_utils.load`
            if "FasterTransformer" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                    decoding_lib)
                LOADED_EXT["FasterTransformer"] = ops
        else:
            if decoding_lib is not None:
                logger.warning(
                    "The specified decoding_lib does not exist, and it will be built automatically."
                )
            load("FasterTransformer", verbose=True)

        size_per_head = d_model / n_head
        # fuse_qkv can only support size_per_head is one of [32, 64, 128].
        if size_per_head in [32, 64, 128]:
            self._fuse_qkv = True
        else:
            self._fuse_qkv = False

        super(InferTransformerDecoding, self).__init__()
        for arg, value in locals().items():
            if arg not in [
                    "self", "decoder", "word_embedding", "positional_embedding",
                    "linear"
            ]:
                setattr(self, "_" + arg, value)
        # process weights
        if use_fp16_decoding:
            for mod in decoder.layers:
                mod.norm1.weight = transfer_param(mod.norm1.weight)
                mod.norm1.bias = transfer_param(mod.norm1.bias, is_bias=True)
                mod.self_attn.q_proj.weight = transfer_param(
                    mod.self_attn.q_proj.weight)
                mod.self_attn.q_proj.bias = transfer_param(
                    mod.self_attn.q_proj.bias, is_bias=True)
                mod.self_attn.k_proj.weight = transfer_param(
                    mod.self_attn.k_proj.weight)
                mod.self_attn.k_proj.bias = transfer_param(
                    mod.self_attn.k_proj.bias, is_bias=True)
                mod.self_attn.v_proj.weight = transfer_param(
                    mod.self_attn.v_proj.weight)
                mod.self_attn.v_proj.bias = transfer_param(
                    mod.self_attn.v_proj.bias, is_bias=True)
                mod.self_attn.out_proj.weight = transfer_param(
                    mod.self_attn.out_proj.weight)
                mod.self_attn.out_proj.bias = transfer_param(
                    mod.self_attn.out_proj.bias, is_bias=True)

                mod.norm2.weight = transfer_param(mod.norm2.weight)
                mod.norm2.bias = transfer_param(mod.norm2.bias, is_bias=True)
                mod.cross_attn.q_proj.weight = transfer_param(
                    mod.cross_attn.q_proj.weight)
                mod.cross_attn.q_proj.bias = transfer_param(
                    mod.cross_attn.q_proj.bias, is_bias=True)
                mod.cross_attn.k_proj.weight = transfer_param(
                    mod.cross_attn.k_proj.weight)
                mod.cross_attn.k_proj.bias = transfer_param(
                    mod.cross_attn.k_proj.bias, is_bias=True)
                mod.cross_attn.v_proj.weight = transfer_param(
                    mod.cross_attn.v_proj.weight)
                mod.cross_attn.v_proj.bias = transfer_param(
                    mod.cross_attn.v_proj.bias, is_bias=True)
                mod.cross_attn.out_proj.weight = transfer_param(
                    mod.cross_attn.out_proj.weight)
                mod.cross_attn.out_proj.bias = transfer_param(
                    mod.cross_attn.out_proj.bias, is_bias=True)

                mod.norm3.weight = transfer_param(mod.norm3.weight)
                mod.norm3.bias = transfer_param(mod.norm3.bias, is_bias=True)
                mod.linear1.weight = transfer_param(mod.linear1.weight)
                mod.linear1.bias = transfer_param(
                    mod.linear1.bias, is_bias=True)
                mod.linear2.weight = transfer_param(mod.linear2.weight)
                mod.linear2.bias = transfer_param(
                    mod.linear2.bias, is_bias=True)

            decoder.norm.weight = transfer_param(decoder.norm.weight)
            decoder.norm.bias = transfer_param(decoder.norm.bias, is_bias=True)

            linear.weight = transfer_param(linear.weight)
            linear.bias = transfer_param(linear.bias, is_bias=True)

            positional_embedding.weight = transfer_param(
                positional_embedding.weight)
            word_embedding.weight = transfer_param(word_embedding.weight)

        self.slf_ln_weight = []
        self.slf_ln_bias = []
        self.slf_q_weight = []
        self.slf_q_bias = []
        self.slf_k_weight = []
        self.slf_k_bias = []
        self.slf_v_weight = []
        self.slf_v_bias = []
        self.slf_out_weight = []
        self.slf_out_bias = []

        self.cross_ln_weight = []
        self.cross_ln_bias = []
        self.cross_q_weight = []
        self.cross_q_bias = []
        self.cross_k_weight = []
        self.cross_k_bias = []
        self.cross_v_weight = []
        self.cross_v_bias = []
        self.cross_out_weight = []
        self.cross_out_bias = []

        self.ffn_ln_weight = []
        self.ffn_ln_bias = []
        self.ffn_inter_weight = []
        self.ffn_inter_bias = []
        self.ffn_out_weight = []
        self.ffn_out_bias = []

        for i, mod in enumerate(decoder.layers):
            self.slf_ln_weight.append(mod.norm1.weight)
            self.slf_ln_bias.append(mod.norm1.bias)

            if self._fuse_qkv:
                q_weight_shape = mod.self_attn.q_proj.weight.shape
                k_weight_shape = mod.self_attn.k_proj.weight.shape
                v_weight_shape = mod.self_attn.v_proj.weight.shape

                q_weights = self.create_parameter(
                    shape=[
                        q_weight_shape[0], q_weight_shape[1] + k_weight_shape[1]
                        + v_weight_shape[1]
                    ],
                    dtype="float16" if use_fp16_decoding else "float32")
                setattr(self, "slf_q_weight_" + str(i), q_weights)
                self.slf_q_weight.append(
                    getattr(self, "slf_q_weight_" + str(i)))

                q_bias_shape = mod.self_attn.q_proj.bias.shape
                k_bias_shape = mod.self_attn.k_proj.bias.shape
                v_bias_shape = mod.self_attn.v_proj.bias.shape

                q_biases = self.create_parameter(
                    shape=[
                        q_bias_shape[0] + k_bias_shape[0] + v_bias_shape[0]
                    ],
                    dtype="float16" if use_fp16_decoding else "float32",
                    is_bias=True)
                setattr(self, "slf_q_bias_" + str(i), q_biases)
                self.slf_q_bias.append(getattr(self, "slf_q_bias_" + str(i)))
            else:
                self.slf_q_weight.append(mod.self_attn.q_proj.weight)
                self.slf_q_bias.append(mod.self_attn.q_proj.bias)

            self.slf_k_weight.append(mod.self_attn.k_proj.weight)
            self.slf_k_bias.append(mod.self_attn.k_proj.bias)
            self.slf_v_weight.append(mod.self_attn.v_proj.weight)
            self.slf_v_bias.append(mod.self_attn.v_proj.bias)
            self.slf_out_weight.append(mod.self_attn.out_proj.weight)
            self.slf_out_bias.append(mod.self_attn.out_proj.bias)

            self.cross_ln_weight.append(mod.norm2.weight)
            self.cross_ln_bias.append(mod.norm2.bias)
            self.cross_q_weight.append(mod.cross_attn.q_proj.weight)
            self.cross_q_bias.append(mod.cross_attn.q_proj.bias)
            self.cross_k_weight.append(mod.cross_attn.k_proj.weight)
            self.cross_k_bias.append(mod.cross_attn.k_proj.bias)
            self.cross_v_weight.append(mod.cross_attn.v_proj.weight)
            self.cross_v_bias.append(mod.cross_attn.v_proj.bias)
            self.cross_out_weight.append(mod.cross_attn.out_proj.weight)
            self.cross_out_bias.append(mod.cross_attn.out_proj.bias)

            self.ffn_ln_weight.append(mod.norm3.weight)
            self.ffn_ln_bias.append(mod.norm3.bias)
            self.ffn_inter_weight.append(mod.linear1.weight)
            self.ffn_inter_bias.append(mod.linear1.bias)
            self.ffn_out_weight.append(mod.linear2.weight)
            self.ffn_out_bias.append(mod.linear2.bias)

        self.decoder_ln_weight = [decoder.norm.weight]
        self.decoder_ln_bias = [decoder.norm.bias]

        self.pos_emb = [positional_embedding.weight]
        self.word_emb = [word_embedding.weight]

        self.linear_weight = [linear.weight]
        self.linear_bias = [linear.bias]

    def forward(self, enc_output, memory_seq_lens, trg_word=None):
        def parse_function(func_name):
            return partial(
                func_name,
                word_emb=self.word_emb,
                slf_ln_weight=self.slf_ln_weight,
                slf_ln_bias=self.slf_ln_bias,
                slf_q_weight=self.slf_q_weight,
                slf_q_bias=self.slf_q_bias,
                slf_k_weight=self.slf_k_weight,
                slf_k_bias=self.slf_k_bias,
                slf_v_weight=self.slf_v_weight,
                slf_v_bias=self.slf_v_bias,
                slf_out_weight=self.slf_out_weight,
                slf_out_bias=self.slf_out_bias,
                cross_ln_weight=self.cross_ln_weight,
                cross_ln_bias=self.cross_ln_bias,
                cross_q_weight=self.cross_q_weight,
                cross_q_bias=self.cross_q_bias,
                cross_k_weight=self.cross_k_weight,
                cross_k_bias=self.cross_k_bias,
                cross_v_weight=self.cross_v_weight,
                cross_v_bias=self.cross_v_bias,
                cross_out_weight=self.cross_out_weight,
                cross_out_bias=self.cross_out_bias,
                ffn_ln_weight=self.ffn_ln_weight,
                ffn_ln_bias=self.ffn_ln_bias,
                ffn_inter_weight=self.ffn_inter_weight,
                ffn_inter_bias=self.ffn_inter_bias,
                ffn_out_weight=self.ffn_out_weight,
                ffn_out_bias=self.ffn_out_bias,
                decoder_ln_weight=self.decoder_ln_weight,
                decoder_ln_bias=self.decoder_ln_bias,
                linear_weight=self.linear_weight,
                linear_bias=self.linear_bias,
                pos_emb=self.pos_emb,
                _decoding_strategy=self._decoding_strategy,
                _beam_size=self._beam_size,
                _topk=self._topk,
                _topp=self._topp,
                _n_head=self._n_head,
                _size_per_head=int(self._d_model / self._n_head),
                _n_layer=self._num_decoder_layers,
                _bos_id=self._bos_id,
                _eos_id=self._eos_id,
                _max_out_len=self._max_out_len,
                _diversity_rate=self._diversity_rate,
                _rel_len=self._rel_len,
                _alpha=self._alpha)

        if self._decoding_strategy.startswith("beam_search"):
            # TODO: Due to paddle.tile bug in static graph, tile_beam_merge_with_batch
            # cannot work properly. These comments should be opened after PaddlePaddle v2.2.2.
            if paddle.__version__ <= "2.1.3":
                enc_output = nn.decode.BeamSearchDecoder.tile_beam_merge_with_batch(
                    enc_output, self._beam_size)
                memory_seq_lens = nn.decode.BeamSearchDecoder.tile_beam_merge_with_batch(
                    memory_seq_lens, self._beam_size)
            else:
                enc_output_shape = paddle.shape(enc_output)
                batch_size = enc_output_shape[0]
                max_seq_len = enc_output_shape[1]
                enc_output = enc_output.unsqueeze([1])
                memory_seq_lens = memory_seq_lens.unsqueeze([1])
                enc_output = paddle.expand(
                    enc_output,
                    shape=[
                        batch_size, self._beam_size, max_seq_len, self._d_model
                    ]
                ).reshape(
                    [batch_size * self._beam_size, max_seq_len, self._d_model])
                memory_seq_lens = paddle.expand(
                    memory_seq_lens,
                    shape=[batch_size, self._beam_size]).reshape(
                        [batch_size * self._beam_size])

        if trg_word is None:
            output_ids, parent_ids, sequence_length = parse_function(
                infer_transformer_decoding)(enc_output=[enc_output],
                                            memory_seq_lens=[memory_seq_lens])
        else:
            output_ids, parent_ids, sequence_length = parse_function(
                infer_force_decoding)(enc_output=[enc_output],
                                      memory_seq_lens=[memory_seq_lens],
                                      trg_word=[trg_word])

        ids = finalize(
            self._beam_size,
            output_ids,
            parent_ids,
            sequence_length,
            decoding_strategy=self._decoding_strategy)

        return ids


class InferGptDecoding(nn.Layer):
    def __init__(self, model, decoding_lib=None, use_fp16_decoding=False):
        if decoding_lib is not None and os.path.isfile(decoding_lib):
            if "FasterTransformer" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                    decoding_lib)
                LOADED_EXT["FasterTransformer"] = ops
        else:
            if decoding_lib is not None:
                logger.warning(
                    "The specified decoding_lib does not exist, and it will be built automatically."
                )
            load("FasterTransformer", verbose=True)

        super(InferGptDecoding, self).__init__()

        self.use_fp16_decoding = use_fp16_decoding
        self.model = model
        self.head_num = self.model.gpt.config['num_attention_heads']
        self.size_per_head = int(self.model.gpt.config['hidden_size'] /
                                 self.head_num)
        self.num_layer = self.model.gpt.config['num_hidden_layers']

        if self.use_fp16_decoding:
            for mod in self.model.gpt.decoder.layers:
                mod.norm1.weight = transfer_param(
                    mod.norm1.weight, restore_data=True)
                mod.norm1.bias = transfer_param(
                    mod.norm1.bias, is_bias=True, restore_data=True)
                mod.self_attn.q_proj.weight = transfer_param(
                    mod.self_attn.q_proj.weight, restore_data=True)
                mod.self_attn.q_proj.bias = transfer_param(
                    mod.self_attn.q_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.k_proj.weight = transfer_param(
                    mod.self_attn.k_proj.weight, restore_data=True)
                mod.self_attn.k_proj.bias = transfer_param(
                    mod.self_attn.k_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.v_proj.weight = transfer_param(
                    mod.self_attn.v_proj.weight, restore_data=True)
                mod.self_attn.v_proj.bias = transfer_param(
                    mod.self_attn.v_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.out_proj.weight = transfer_param(
                    mod.self_attn.out_proj.weight, restore_data=True)
                mod.self_attn.out_proj.bias = transfer_param(
                    mod.self_attn.out_proj.bias,
                    is_bias=True,
                    restore_data=True)

                mod.norm2.weight = transfer_param(
                    mod.norm2.weight, restore_data=True)
                mod.norm2.bias = transfer_param(
                    mod.norm2.bias, is_bias=True, restore_data=True)
                mod.linear1.weight = transfer_param(
                    mod.linear1.weight, restore_data=True)
                mod.linear1.bias = transfer_param(
                    mod.linear1.bias, is_bias=True, restore_data=True)
                mod.linear2.weight = transfer_param(
                    mod.linear2.weight, restore_data=True)
                mod.linear2.bias = transfer_param(
                    mod.linear2.bias, is_bias=True, restore_data=True)

            self.model.gpt.embeddings.word_embeddings.weight = transfer_param(
                self.model.gpt.embeddings.word_embeddings.weight,
                restore_data=True)
            self.model.gpt.embeddings.position_embeddings.weight = transfer_param(
                self.model.gpt.embeddings.position_embeddings.weight,
                restore_data=True)
            self.model.gpt.decoder.norm.weight = transfer_param(
                self.model.gpt.decoder.norm.weight, restore_data=True)
            self.model.gpt.decoder.norm.bias = transfer_param(
                self.model.gpt.decoder.norm.bias, restore_data=True)

        self.linear_weight = [self.model.gpt.embeddings.word_embeddings.weight]

        self.slf_ln_weight = []
        self.slf_ln_bias = []
        self.slf_q_weight = []
        self.slf_q_bias = []
        self.slf_k_weight = []
        self.slf_k_bias = []
        self.slf_v_weight = []
        self.slf_v_bias = []
        self.slf_out_weight = []
        self.slf_out_bias = []

        self.ffn_ln_weight = []
        self.ffn_ln_bias = []
        self.ffn_inter_weight = []
        self.ffn_inter_bias = []
        self.ffn_out_weight = []
        self.ffn_out_bias = []

        for i, mod in enumerate(self.model.gpt.decoder.layers):
            self.slf_ln_weight.append(mod.norm1.weight)
            self.slf_ln_bias.append(mod.norm1.bias)

            q_weights = paddle.concat(
                [
                    mod.self_attn.q_proj.weight, mod.self_attn.k_proj.weight,
                    mod.self_attn.v_proj.weight
                ],
                axis=-1)
            setattr(self, "slf_q_weight_" + str(i), q_weights)
            self.slf_q_weight.append(getattr(self, "slf_q_weight_" + str(i)))

            q_biases = paddle.concat(
                [
                    mod.self_attn.q_proj.bias, mod.self_attn.k_proj.bias,
                    mod.self_attn.v_proj.bias
                ],
                axis=-1)
            setattr(self, "slf_q_bias_" + str(i), q_biases)
            self.slf_q_bias.append(getattr(self, "slf_q_bias_" + str(i)))

            self.slf_k_weight.append(mod.self_attn.k_proj.weight)
            self.slf_k_bias.append(mod.self_attn.k_proj.bias)
            self.slf_v_weight.append(mod.self_attn.v_proj.weight)
            self.slf_v_bias.append(mod.self_attn.v_proj.bias)
            self.slf_out_weight.append(mod.self_attn.out_proj.weight)
            self.slf_out_bias.append(mod.self_attn.out_proj.bias)

            self.ffn_ln_weight.append(mod.norm2.weight)
            self.ffn_ln_bias.append(mod.norm2.bias)
            self.ffn_inter_weight.append(mod.linear1.weight)
            self.ffn_inter_bias.append(mod.linear1.bias)
            self.ffn_out_weight.append(mod.linear2.weight)
            self.ffn_out_bias.append(mod.linear2.bias)

        self.decoder_ln_weight = [self.model.gpt.decoder.norm.weight]
        self.decoder_ln_bias = [self.model.gpt.decoder.norm.bias]

        self.pos_emb = [self.model.gpt.embeddings.position_embeddings.weight]
        self.word_emb = [self.model.gpt.embeddings.word_embeddings.weight]

    def forward(self,
                input_ids,
                mem_seq_len,
                attention_mask=None,
                topk=4,
                topp=0.0,
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
                forced_eos_token_id=None,
                max_out_len=256,
                temperature=1):
        if attention_mask is None:
            batch_size = paddle.shape(input_ids)[0]
            attention_mask = paddle.tril(
                paddle.ones(
                    [batch_size, mem_seq_len, mem_seq_len],
                    dtype="float16" if self.use_fp16_decoding else "float32"))
        elif self.use_fp16_decoding and attention_mask.dtype == paddle.float32:
            attention_mask = paddle.cast(attention_mask, dtype="float16")

        output_ids = infer_gpt_decoding(
            input=[input_ids],
            attn_mask=[attention_mask],
            mem_seq_len=[mem_seq_len],
            word_emb=self.word_emb,
            slf_ln_weight=self.slf_ln_weight,
            slf_ln_bias=self.slf_ln_bias,
            slf_q_weight=self.slf_q_weight,
            slf_q_bias=self.slf_q_bias,
            slf_k_weight=self.slf_k_weight,
            slf_k_bias=self.slf_k_bias,
            slf_v_weight=self.slf_v_weight,
            slf_v_bias=self.slf_v_bias,
            slf_out_weight=self.slf_out_weight,
            slf_out_bias=self.slf_out_bias,
            ffn_ln_weight=self.ffn_ln_weight,
            ffn_ln_bias=self.ffn_ln_bias,
            ffn_inter_weight=self.ffn_inter_weight,
            ffn_inter_bias=self.ffn_inter_bias,
            ffn_out_weight=self.ffn_out_weight,
            ffn_out_bias=self.ffn_out_bias,
            decoder_ln_weight=self.decoder_ln_weight,
            decoder_ln_bias=self.decoder_ln_bias,
            pos_emb=self.pos_emb,
            linear_weight=self.linear_weight,
            topk=topk,
            topp=topp,
            max_out_len=max_out_len,
            head_num=self.head_num,
            size_per_head=self.size_per_head,
            num_layer=self.num_layer,
            bos_id=bos_token_id,
            eos_id=eos_token_id,
            temperature=temperature,
            use_fp16_decoding=self.use_fp16_decoding)

        output_ids = output_ids[paddle.shape(input_ids)[-1]:, :]
        if forced_eos_token_id is not None:
            output_ids[:, -1] = forced_eos_token_id
        return output_ids


class InferUnifiedDecoding(nn.Layer):
    def __init__(self,
                 model,
                 decoding_lib=None,
                 use_fp16_decoding=False,
                 logits_mask=None,
                 n_head=8,
                 hidden_dims=512,
                 size_per_head=64,
                 n_layer=6,
                 unk_id=0,
                 mask_id=30000,
                 normalize_before=True,
                 hidden_act="gelu"):
        if decoding_lib is not None and os.path.isfile(decoding_lib):
            # Maybe it has been loadad by `ext_utils.load`
            if "FasterTransformer" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                    decoding_lib)
                LOADED_EXT["FasterTransformer"] = ops
        else:
            if decoding_lib is not None:
                logger.warning(
                    "The specified decoding_lib does not exist, and it will be built automatically."
                )
            load("FasterTransformer", verbose=True)

        super(InferUnifiedDecoding, self).__init__()
        for arg, value in locals().items():
            if arg not in ["self"]:
                setattr(self, "_" + arg, value)

        self.sub_modules = {
            "slf_ln_weight": [],
            "slf_ln_bias": [],
            "slf_q_weight": [],
            "slf_q_bias": [],
            "slf_k_weight": [],
            "slf_k_bias": [],
            "slf_v_weight": [],
            "slf_v_bias": [],
            "slf_out_weight": [],
            "slf_out_bias": [],
            "ffn_ln_weight": [],
            "ffn_ln_bias": [],
            "ffn_inter_weight": [],
            "ffn_inter_bias": [],
            "ffn_out_weight": [],
            "ffn_out_bias": [],
            "word_emb": None,
            "pos_emb": None,
            "type_emb": None,
            "decoder_ln_weight": None,
            "decoder_ln_bias": None,
            "trans_weight": None,
            "trans_bias": None,
            "lm_ln_weight": None,
            "lm_ln_bias": None,
            "linear_weight": None,
            "linear_bias": None
        }
        if self._use_fp16_decoding:
            for mod in self._model.encoder.layers:
                self.sub_modules["slf_q_weight"].append(
                    paddle.concat(
                        [
                            transfer_param(
                                mod.self_attn.q_proj.weight, restore_data=True),
                            transfer_param(
                                mod.self_attn.k_proj.weight, restore_data=True),
                            transfer_param(
                                mod.self_attn.v_proj.weight, restore_data=True)
                        ],
                        axis=-1))
                self.sub_modules["slf_q_bias"].append(
                    paddle.concat(
                        [
                            transfer_param(
                                mod.self_attn.q_proj.bias,
                                is_bias=True,
                                restore_data=True), transfer_param(
                                    mod.self_attn.k_proj.bias,
                                    is_bias=True,
                                    restore_data=True), transfer_param(
                                        mod.self_attn.v_proj.bias,
                                        is_bias=True,
                                        restore_data=True)
                        ],
                        axis=-1))
                self.sub_modules["slf_k_weight"].append(
                    transfer_param(
                        mod.self_attn.k_proj.weight, restore_data=True))
                self.sub_modules["slf_k_bias"].append(
                    transfer_param(
                        mod.self_attn.k_proj.bias,
                        is_bias=True,
                        restore_data=True))
                self.sub_modules["slf_v_weight"].append(
                    transfer_param(
                        mod.self_attn.v_proj.weight, restore_data=True))
                self.sub_modules["slf_v_bias"].append(
                    transfer_param(
                        mod.self_attn.v_proj.bias,
                        is_bias=True,
                        restore_data=True))
                self.sub_modules["slf_out_weight"].append(
                    transfer_param(
                        mod.self_attn.out_proj.weight, restore_data=True))
                self.sub_modules["slf_out_bias"].append(
                    transfer_param(
                        mod.self_attn.out_proj.bias,
                        is_bias=True,
                        restore_data=True))
                self.sub_modules["ffn_inter_weight"].append(
                    transfer_param(
                        mod.linear1.weight, restore_data=True))
                self.sub_modules["ffn_inter_bias"].append(
                    transfer_param(
                        mod.linear1.bias, is_bias=True, restore_data=True))
                self.sub_modules["ffn_out_weight"].append(
                    transfer_param(
                        mod.linear2.weight, restore_data=True))
                self.sub_modules["ffn_out_bias"].append(
                    transfer_param(
                        mod.linear2.bias, is_bias=True, restore_data=True))
                self.sub_modules["slf_ln_weight"].append(
                    transfer_param(
                        mod.norm1.weight, restore_data=True))
                self.sub_modules["slf_ln_bias"].append(
                    transfer_param(
                        mod.norm1.bias, is_bias=True, restore_data=True))
                self.sub_modules["ffn_ln_weight"].append(
                    transfer_param(
                        mod.norm2.weight, restore_data=True))
                self.sub_modules["ffn_ln_bias"].append(
                    transfer_param(
                        mod.norm2.bias, is_bias=True, restore_data=True))

            self.sub_modules["word_emb"] = [
                transfer_param(
                    self._model.embeddings.word_embeddings.weight,
                    restore_data=True)
            ]
            self.sub_modules["pos_emb"] = [
                transfer_param(
                    self._model.embeddings.position_embeddings.weight,
                    restore_data=True)
            ]
            self.sub_modules["type_emb"] = [
                transfer_param(
                    self._model.embeddings.token_type_embeddings.weight,
                    restore_data=True)
            ]
            if self._normalize_before:
                self.sub_modules["decoder_ln_weight"] = [
                    transfer_param(
                        self._model.encoder.norm.weight, restore_data=True)
                ]
                self.sub_modules["decoder_ln_bias"] = [
                    transfer_param(
                        self._model.encoder.norm.bias,
                        is_bias=True,
                        restore_data=True)
                ]
            else:
                self.sub_modules["decoder_ln_weight"] = [
                    transfer_param(
                        self._model.encoder_norm.weight, restore_data=True)
                ]
                self.sub_modules["decoder_ln_bias"] = [
                    transfer_param(
                        self._model.encoder_norm.bias,
                        is_bias=True,
                        restore_data=True)
                ]
            self.sub_modules["trans_weight"] = [
                transfer_param(
                    self._model.lm_head.transform.weight, restore_data=True)
            ]
            self.sub_modules["trans_bias"] = [
                transfer_param(
                    self._model.lm_head.transform.bias,
                    is_bias=True,
                    restore_data=True)
            ]
            self.sub_modules["lm_ln_weight"] = [
                transfer_param(
                    self._model.lm_head.layer_norm.weight, restore_data=True)
            ]
            self.sub_modules["lm_ln_bias"] = [
                transfer_param(
                    self._model.lm_head.layer_norm.bias,
                    is_bias=True,
                    restore_data=True)
            ]
            self.sub_modules["linear_weight"] = [
                paddle.transpose(
                    transfer_param(
                        self._model.lm_head.decoder_weight, restore_data=True),
                    [1, 0])
            ]
            self.sub_modules["linear_bias"] = [
                transfer_param(
                    self._model.lm_head.decoder_bias,
                    is_bias=True,
                    restore_data=True)
            ]
        else:
            for mod in self._model.encoder.layers:
                self.sub_modules["slf_q_weight"].append(
                    paddle.concat(
                        [
                            mod.self_attn.q_proj.weight, mod.self_attn.k_proj.
                            weight, mod.self_attn.v_proj.weight
                        ],
                        axis=-1))
                self.sub_modules["slf_q_bias"].append(
                    paddle.concat(
                        [
                            mod.self_attn.q_proj.bias,
                            mod.self_attn.k_proj.bias, mod.self_attn.v_proj.bias
                        ],
                        axis=-1))
                self.sub_modules["slf_k_weight"].append(
                    mod.self_attn.k_proj.weight)
                self.sub_modules["slf_k_bias"].append(mod.self_attn.k_proj.bias)
                self.sub_modules["slf_v_weight"].append(
                    mod.self_attn.v_proj.weight)
                self.sub_modules["slf_v_bias"].append(mod.self_attn.v_proj.bias)
                self.sub_modules["slf_out_weight"].append(
                    mod.self_attn.out_proj.weight)
                self.sub_modules["slf_out_bias"].append(
                    mod.self_attn.out_proj.bias)
                self.sub_modules["ffn_inter_weight"].append(mod.linear1.weight)
                self.sub_modules["ffn_inter_bias"].append(mod.linear1.bias)
                self.sub_modules["ffn_out_weight"].append(mod.linear2.weight)
                self.sub_modules["ffn_out_bias"].append(mod.linear2.bias)
                self.sub_modules["slf_ln_weight"].append(mod.norm1.weight)
                self.sub_modules["slf_ln_bias"].append(mod.norm1.bias)
                self.sub_modules["ffn_ln_weight"].append(mod.norm2.weight)
                self.sub_modules["ffn_ln_bias"].append(mod.norm2.bias)

            self.sub_modules[
                "word_emb"] = [self._model.embeddings.word_embeddings.weight]
            self.sub_modules["pos_emb"] = [
                self._model.embeddings.position_embeddings.weight
            ]
            self.sub_modules["type_emb"] = [
                self._model.embeddings.token_type_embeddings.weight
            ]
            if self._normalize_before:
                self.sub_modules[
                    "decoder_ln_weight"] = [self._model.encoder.norm.weight]
                self.sub_modules[
                    "decoder_ln_bias"] = [self._model.encoder.norm.bias]
            else:
                self.sub_modules[
                    "decoder_ln_weight"] = [self._model.encoder_norm.weight]
                self.sub_modules[
                    "decoder_ln_bias"] = [self._model.encoder_norm.bias]

            self.sub_modules[
                "trans_weight"] = [self._model.lm_head.transform.weight]
            self.sub_modules[
                "trans_bias"] = [self._model.lm_head.transform.bias]
            self.sub_modules[
                "lm_ln_weight"] = [self._model.lm_head.layer_norm.weight]
            self.sub_modules[
                "lm_ln_bias"] = [self._model.lm_head.layer_norm.bias]
            self.sub_modules[
                "linear_weight"] = [self._model.lm_head.decoder_weight.t()]

            # NOTE: Fix self._model.lm_head.decoder_bias been changed in FT.
            self.sub_modules["linear_bias"] = [
                paddle.assign(self._model.lm_head.decoder_bias)
            ]

    def forward(self,
                cache_k,
                cache_v,
                memory_seq_lens,
                decoding_type_id,
                beam_size=4,
                topk=4,
                topp=0.0,
                decoding_strategy="greedy_search",
                max_out_len=256,
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
                forced_eos_token_id=None,
                temperature=1.0,
                length_penalty=1.0,
                diversity_rate=0.0,
                pos_bias=True,
                rel_len=False,
                early_stopping=False):
        if decoding_strategy == "greedy_search":
            decoding_strategy = "topk_sampling"
            topk = 1
            topp = 0
        elif decoding_strategy in [
                "sampling", "topk_sampling", "topp_sampling"
        ]:
            if topp == 1 and topk > 0:
                decoding_strategy = "topk_sampling"
                topp = 0
            elif topp > 0 and topk == 0:
                decoding_strategy = "topp_sampling"
            else:
                raise AttributeError(
                    "Only topk sampling or topp sampling are supported. " \
                    "Topk sampling and topp sampling cannot be both applied in the faster version.")
        elif decoding_strategy.startswith("beam_search"):
            decoding_strategy = "beam_search_v3"
        output_ids, parent_ids, sequence_length = infer_unified_decoding(
            cache_k=cache_k,
            cache_v=cache_v,
            memory_seq_lens=[memory_seq_lens],
            type_id=[decoding_type_id],
            logits_mask=[self._logits_mask],
            word_emb=self.sub_modules["word_emb"],
            slf_ln_weight=self.sub_modules["slf_ln_weight"],
            slf_ln_bias=self.sub_modules["slf_ln_bias"],
            slf_q_weight=self.sub_modules["slf_q_weight"],
            slf_q_bias=self.sub_modules["slf_q_bias"],
            slf_k_weight=self.sub_modules["slf_k_weight"],
            slf_k_bias=self.sub_modules["slf_k_bias"],
            slf_v_weight=self.sub_modules["slf_v_weight"],
            slf_v_bias=self.sub_modules["slf_v_bias"],
            slf_out_weight=self.sub_modules["slf_out_weight"],
            slf_out_bias=self.sub_modules["slf_out_bias"],
            ffn_ln_weight=self.sub_modules["ffn_ln_weight"],
            ffn_ln_bias=self.sub_modules["ffn_ln_bias"],
            ffn_inter_weight=self.sub_modules["ffn_inter_weight"],
            ffn_inter_bias=self.sub_modules["ffn_inter_bias"],
            ffn_out_weight=self.sub_modules["ffn_out_weight"],
            ffn_out_bias=self.sub_modules["ffn_out_bias"],
            decoder_ln_weight=self.sub_modules["decoder_ln_weight"],
            decoder_ln_bias=self.sub_modules["decoder_ln_bias"],
            trans_weight=self.sub_modules["trans_weight"],
            trans_bias=self.sub_modules["trans_bias"],
            lm_ln_weight=self.sub_modules["lm_ln_weight"],
            lm_ln_bias=self.sub_modules["lm_ln_bias"],
            linear_weight=self.sub_modules["linear_weight"],
            linear_bias=self.sub_modules["linear_bias"],
            pos_emb=self.sub_modules["pos_emb"],
            type_emb=self.sub_modules["type_emb"],
            _decoding_strategy=decoding_strategy,
            _beam_size=beam_size,
            _topk=topk,
            _topp=topp,
            _n_head=self._n_head,
            _size_per_head=self._size_per_head,
            _n_layer=self._n_layer,
            _bos_id=bos_token_id,
            _eos_id=eos_token_id,
            _max_out_len=max_out_len,
            _diversity_rate=-diversity_rate,
            _unk_id=self._unk_id,
            _mask_id=self._mask_id,
            _temperature=temperature,
            _len_penalty=length_penalty,
            _normalize_before=self._normalize_before,
            _pos_bias=pos_bias,
            _hidden_act=self._hidden_act,
            _rel_len=rel_len,
            _early_stopping=early_stopping)
        ids = finalize(
            beam_size,
            output_ids,
            parent_ids,
            sequence_length,
            forced_eos_token_id=forced_eos_token_id,
            decoding_strategy=decoding_strategy)
        return ids


class InferBartDecoding(nn.Layer):
    def __init__(self, model, decoding_lib=None, use_fp16_decoding=False):
        if decoding_lib is not None and os.path.isfile(decoding_lib):
            # Maybe it has been loadad by `ext_utils.load`
            if "FasterTransformer" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                    decoding_lib)
                LOADED_EXT["FasterTransformer"] = ops
        else:
            if decoding_lib is not None:
                logger.warning(
                    "The specified decoding_lib does not exist, and it will be built automatically."
                )
            load("FasterTransformer", verbose=True)

        super(InferBartDecoding, self).__init__()
        for arg, value in locals().items():
            if arg not in [
                    "self", "model", "word_embedding", "positional_embedding",
                    "linear"
            ]:
                setattr(self, "_" + arg, value)
        self._num_decoder_layers = model.bart.config['num_decoder_layers']
        self._n_head = model.bart.config['decoder_attention_heads']
        self._d_model = model.bart.config['d_model']

        # process weights
        if use_fp16_decoding:
            for mod in model.bart.decoder.decoder.layers:
                mod.norm1.weight = transfer_param(
                    mod.norm1.weight, restore_data=True)
                mod.norm1.bias = transfer_param(
                    mod.norm1.bias, is_bias=True, restore_data=True)
                mod.self_attn.q_proj.weight = transfer_param(
                    mod.self_attn.q_proj.weight, restore_data=True)
                mod.self_attn.q_proj.bias = transfer_param(
                    mod.self_attn.q_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.k_proj.weight = transfer_param(
                    mod.self_attn.k_proj.weight, restore_data=True)
                mod.self_attn.k_proj.bias = transfer_param(
                    mod.self_attn.k_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.v_proj.weight = transfer_param(
                    mod.self_attn.v_proj.weight, restore_data=True)
                mod.self_attn.v_proj.bias = transfer_param(
                    mod.self_attn.v_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.out_proj.weight = transfer_param(
                    mod.self_attn.out_proj.weight, restore_data=True)
                mod.self_attn.out_proj.bias = transfer_param(
                    mod.self_attn.out_proj.bias,
                    is_bias=True,
                    restore_data=True)

                mod.norm2.weight = transfer_param(
                    mod.norm2.weight, restore_data=True)
                mod.norm2.bias = transfer_param(
                    mod.norm2.bias, is_bias=True, restore_data=True)
                mod.cross_attn.q_proj.weight = transfer_param(
                    mod.cross_attn.q_proj.weight, restore_data=True)
                mod.cross_attn.q_proj.bias = transfer_param(
                    mod.cross_attn.q_proj.bias, is_bias=True, restore_data=True)
                mod.cross_attn.k_proj.weight = transfer_param(
                    mod.cross_attn.k_proj.weight, restore_data=True)
                mod.cross_attn.k_proj.bias = transfer_param(
                    mod.cross_attn.k_proj.bias, is_bias=True, restore_data=True)
                mod.cross_attn.v_proj.weight = transfer_param(
                    mod.cross_attn.v_proj.weight, restore_data=True)
                mod.cross_attn.v_proj.bias = transfer_param(
                    mod.cross_attn.v_proj.bias, is_bias=True, restore_data=True)
                mod.cross_attn.out_proj.weight = transfer_param(
                    mod.cross_attn.out_proj.weight, restore_data=True)
                mod.cross_attn.out_proj.bias = transfer_param(
                    mod.cross_attn.out_proj.bias,
                    is_bias=True,
                    restore_data=True)

                mod.norm3.weight = transfer_param(
                    mod.norm3.weight, restore_data=True)
                mod.norm3.bias = transfer_param(
                    mod.norm3.bias, is_bias=True, restore_data=True)
                mod.linear1.weight = transfer_param(
                    mod.linear1.weight, restore_data=True)
                mod.linear1.bias = transfer_param(
                    mod.linear1.bias, is_bias=True, restore_data=True)
                mod.linear2.weight = transfer_param(
                    mod.linear2.weight, restore_data=True)
                mod.linear2.bias = transfer_param(
                    mod.linear2.bias, is_bias=True, restore_data=True)

            model.decoder.decoder_layernorm_embedding.weight = transfer_param(
                model.decoder.decoder_layernorm_embedding.weight,
                restore_data=True)
            model.decoder.decoder_layernorm_embedding.bias = transfer_param(
                model.decoder.decoder_layernorm_embedding.bias,
                is_bias=True,
                restore_data=True)

            model.lm_head_weight = transfer_param(
                model.lm_head_weight, restore_data=True)
            model.final_logits_bias = transfer_param(
                model.final_logits_bias, is_bias=True, restore_data=True)

            model.decoder.decoder_embed_positions.weight = transfer_param(
                model.decoder.decoder_embed_positions.weight, restore_data=True)
            model.decoder.embed_tokens.weight = transfer_param(
                model.decoder.embed_tokens.weight, restore_data=True)

        self.slf_ln_weight = []
        self.slf_ln_bias = []
        self.slf_q_weight = []
        self.slf_q_bias = []
        self.slf_k_weight = []
        self.slf_k_bias = []
        self.slf_v_weight = []
        self.slf_v_bias = []
        self.slf_out_weight = []
        self.slf_out_bias = []

        self.cross_ln_weight = []
        self.cross_ln_bias = []
        self.cross_q_weight = []
        self.cross_q_bias = []
        self.cross_k_weight = []
        self.cross_k_bias = []
        self.cross_v_weight = []
        self.cross_v_bias = []
        self.cross_out_weight = []
        self.cross_out_bias = []

        self.ffn_ln_weight = []
        self.ffn_ln_bias = []
        self.ffn_inter_weight = []
        self.ffn_inter_bias = []
        self.ffn_out_weight = []
        self.ffn_out_bias = []

        for mod in model.bart.decoder.decoder.layers:
            self.slf_ln_weight.append(mod.norm1.weight)
            self.slf_ln_bias.append(mod.norm1.bias)
            self.slf_q_weight.append(mod.self_attn.q_proj.weight)
            self.slf_q_bias.append(mod.self_attn.q_proj.bias)
            self.slf_k_weight.append(mod.self_attn.k_proj.weight)
            self.slf_k_bias.append(mod.self_attn.k_proj.bias)
            self.slf_v_weight.append(mod.self_attn.v_proj.weight)
            self.slf_v_bias.append(mod.self_attn.v_proj.bias)
            self.slf_out_weight.append(mod.self_attn.out_proj.weight)
            self.slf_out_bias.append(mod.self_attn.out_proj.bias)

            self.cross_ln_weight.append(mod.norm2.weight)
            self.cross_ln_bias.append(mod.norm2.bias)
            self.cross_q_weight.append(mod.cross_attn.q_proj.weight)
            self.cross_q_bias.append(mod.cross_attn.q_proj.bias)
            self.cross_k_weight.append(mod.cross_attn.k_proj.weight)
            self.cross_k_bias.append(mod.cross_attn.k_proj.bias)
            self.cross_v_weight.append(mod.cross_attn.v_proj.weight)
            self.cross_v_bias.append(mod.cross_attn.v_proj.bias)
            self.cross_out_weight.append(mod.cross_attn.out_proj.weight)
            self.cross_out_bias.append(mod.cross_attn.out_proj.bias)

            self.ffn_ln_weight.append(mod.norm3.weight)
            self.ffn_ln_bias.append(mod.norm3.bias)
            self.ffn_inter_weight.append(mod.linear1.weight)
            self.ffn_inter_bias.append(mod.linear1.bias)
            self.ffn_out_weight.append(mod.linear2.weight)
            self.ffn_out_bias.append(mod.linear2.bias)

        self.decoder_ln_weight = [
            model.decoder.decoder_layernorm_embedding.weight
        ]
        self.decoder_ln_bias = [model.decoder.decoder_layernorm_embedding.bias]

        self.pos_emb = [model.decoder.decoder_embed_positions.weight]
        self.word_emb = [model.decoder.embed_tokens.weight]

        self.linear_weight = [model.lm_head_weight.t()]
        self.linear_bias = [model.final_logits_bias]

    def forward(self,
                enc_output,
                memory_seq_lens,
                beam_size=4,
                top_k=1,
                top_p=0.0,
                decoding_strategy="beam_search_v3",
                max_out_len=256,
                diversity_rate=0.0,
                rel_len=False,
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
                forced_eos_token_id=None,
                alpha=0.6,
                early_stopping=False):
        # beam_search/beam_search_v2/beam_search_v3 should be corrected to beam_search_v3.
        if decoding_strategy.startswith("beam_search"):
            decoding_strategy = "beam_search_v3"
        elif decoding_strategy == "greedy_search":
            decoding_strategy = "topk_sampling"
            top_k = 1
            top_p = 0.0
        elif decoding_strategy in [
                "sampling", "topk_sampling", "topp_sampling"
        ]:
            if top_p == 1 and top_k > 0:
                decoding_strategy = "topk_sampling"
                top_p = 0.0
            elif top_p > 0 and top_k == 0:
                decoding_strategy = "topp_sampling"
            else:
                raise AttributeError(
                    "Only topk sampling or topp sampling are supported. " \
                    "Topk sampling and topp sampling cannot be both applied in the faster version. ")

        output_ids, parent_ids, sequence_length = infer_bart_decoding(
            [enc_output], [memory_seq_lens], self.word_emb, self.slf_ln_weight,
            self.slf_ln_bias, self.slf_q_weight, self.slf_q_bias,
            self.slf_k_weight, self.slf_k_bias, self.slf_v_weight,
            self.slf_v_bias, self.slf_out_weight, self.slf_out_bias,
            self.cross_ln_weight, self.cross_ln_bias, self.cross_q_weight,
            self.cross_q_bias, self.cross_k_weight, self.cross_k_bias,
            self.cross_v_weight, self.cross_v_bias, self.cross_out_weight,
            self.cross_out_bias, self.ffn_ln_weight, self.ffn_ln_bias,
            self.ffn_inter_weight, self.ffn_inter_bias, self.ffn_out_weight,
            self.ffn_out_bias, self.decoder_ln_weight, self.decoder_ln_bias,
            self.linear_weight, self.linear_bias, self.pos_emb,
            decoding_strategy, beam_size, top_k, top_p, self._n_head,
            int(self._d_model / self._n_head), self._num_decoder_layers,
            bos_token_id, eos_token_id, max_out_len, -diversity_rate, rel_len,
            alpha, early_stopping)

        ids = finalize(
            beam_size,
            output_ids,
            parent_ids,
            sequence_length,
            forced_eos_token_id=forced_eos_token_id,
            decoding_strategy=decoding_strategy)
        return ids


class InferMBartDecoding(nn.Layer):
    def __init__(self,
                 model,
                 decoding_lib=None,
                 use_fp16_decoding=False,
                 hidden_act="gelu"):
        if decoding_lib is not None and os.path.isfile(decoding_lib):
            # Maybe it has been loadad by `ext_utils.load`
            if "FasterTransformer" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                    decoding_lib)
                LOADED_EXT["FasterTransformer"] = ops
        else:
            if decoding_lib is not None:
                logger.warning(
                    "The specified decoding_lib does not exist, and it will be built automatically."
                )
            load("FasterTransformer", verbose=True)

        super(InferMBartDecoding, self).__init__()
        for arg, value in locals().items():
            if arg not in [
                    "self", "model", "word_embedding", "positional_embedding",
                    "linear"
            ]:
                setattr(self, "_" + arg, value)
        self._num_decoder_layers = model.mbart.config['num_decoder_layers']
        self._n_head = model.mbart.config['decoder_attention_heads']
        self._d_model = model.mbart.config['d_model']

        # process weights
        if use_fp16_decoding:
            for mod in model.mbart.decoder.decoder.layers:
                mod.norm1.weight = transfer_param(
                    mod.norm1.weight, restore_data=True)
                mod.norm1.bias = transfer_param(
                    mod.norm1.bias, is_bias=True, restore_data=True)
                mod.self_attn.q_proj.weight = transfer_param(
                    mod.self_attn.q_proj.weight, restore_data=True)
                mod.self_attn.q_proj.bias = transfer_param(
                    mod.self_attn.q_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.k_proj.weight = transfer_param(
                    mod.self_attn.k_proj.weight, restore_data=True)
                mod.self_attn.k_proj.bias = transfer_param(
                    mod.self_attn.k_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.v_proj.weight = transfer_param(
                    mod.self_attn.v_proj.weight, restore_data=True)
                mod.self_attn.v_proj.bias = transfer_param(
                    mod.self_attn.v_proj.bias, is_bias=True, restore_data=True)
                mod.self_attn.out_proj.weight = transfer_param(
                    mod.self_attn.out_proj.weight, restore_data=True)
                mod.self_attn.out_proj.bias = transfer_param(
                    mod.self_attn.out_proj.bias,
                    is_bias=True,
                    restore_data=True)

                mod.norm2.weight = transfer_param(
                    mod.norm2.weight, restore_data=True)
                mod.norm2.bias = transfer_param(
                    mod.norm2.bias, is_bias=True, restore_data=True)
                mod.cross_attn.q_proj.weight = transfer_param(
                    mod.cross_attn.q_proj.weight, restore_data=True)
                mod.cross_attn.q_proj.bias = transfer_param(
                    mod.cross_attn.q_proj.bias, is_bias=True, restore_data=True)
                mod.cross_attn.k_proj.weight = transfer_param(
                    mod.cross_attn.k_proj.weight, restore_data=True)
                mod.cross_attn.k_proj.bias = transfer_param(
                    mod.cross_attn.k_proj.bias, is_bias=True, restore_data=True)
                mod.cross_attn.v_proj.weight = transfer_param(
                    mod.cross_attn.v_proj.weight, restore_data=True)
                mod.cross_attn.v_proj.bias = transfer_param(
                    mod.cross_attn.v_proj.bias, is_bias=True, restore_data=True)
                mod.cross_attn.out_proj.weight = transfer_param(
                    mod.cross_attn.out_proj.weight, restore_data=True)
                mod.cross_attn.out_proj.bias = transfer_param(
                    mod.cross_attn.out_proj.bias,
                    is_bias=True,
                    restore_data=True)

                mod.norm3.weight = transfer_param(
                    mod.norm3.weight, restore_data=True)
                mod.norm3.bias = transfer_param(
                    mod.norm3.bias, is_bias=True, restore_data=True)
                mod.linear1.weight = transfer_param(
                    mod.linear1.weight, restore_data=True)
                mod.linear1.bias = transfer_param(
                    mod.linear1.bias, is_bias=True, restore_data=True)
                mod.linear2.weight = transfer_param(
                    mod.linear2.weight, restore_data=True)
                mod.linear2.bias = transfer_param(
                    mod.linear2.bias, is_bias=True, restore_data=True)

            model.decoder.decoder_layernorm_embedding.weight = transfer_param(
                model.decoder.decoder_layernorm_embedding.weight,
                restore_data=True)
            model.decoder.decoder_layernorm_embedding.bias = transfer_param(
                model.decoder.decoder_layernorm_embedding.bias,
                is_bias=True,
                restore_data=True)

            model.decoder.decoder.norm.weight = transfer_param(
                model.decoder.decoder.norm.weight, restore_data=True)
            model.decoder.decoder.norm.bias = transfer_param(
                model.decoder.decoder.norm.bias,
                is_bias=True,
                restore_data=True)

            model.lm_head_weight = transfer_param(
                model.lm_head_weight, restore_data=True)
            model.final_logits_bias = transfer_param(
                model.final_logits_bias, is_bias=True, restore_data=True)

            model.decoder.decoder_embed_positions.weight = transfer_param(
                model.decoder.decoder_embed_positions.weight, restore_data=True)
            model.decoder.embed_tokens.weight = transfer_param(
                model.decoder.embed_tokens.weight, restore_data=True)

        self.slf_ln_weight = []
        self.slf_ln_bias = []
        self.slf_q_weight = []
        self.slf_q_bias = []
        self.slf_k_weight = []
        self.slf_k_bias = []
        self.slf_v_weight = []
        self.slf_v_bias = []
        self.slf_out_weight = []
        self.slf_out_bias = []

        self.cross_ln_weight = []
        self.cross_ln_bias = []
        self.cross_q_weight = []
        self.cross_q_bias = []
        self.cross_k_weight = []
        self.cross_k_bias = []
        self.cross_v_weight = []
        self.cross_v_bias = []
        self.cross_out_weight = []
        self.cross_out_bias = []

        self.ffn_ln_weight = []
        self.ffn_ln_bias = []
        self.ffn_inter_weight = []
        self.ffn_inter_bias = []
        self.ffn_out_weight = []
        self.ffn_out_bias = []

        for mod in model.mbart.decoder.decoder.layers:
            self.slf_ln_weight.append(mod.norm1.weight)
            self.slf_ln_bias.append(mod.norm1.bias)
            self.slf_q_weight.append(mod.self_attn.q_proj.weight)
            self.slf_q_bias.append(mod.self_attn.q_proj.bias)
            self.slf_k_weight.append(mod.self_attn.k_proj.weight)
            self.slf_k_bias.append(mod.self_attn.k_proj.bias)
            self.slf_v_weight.append(mod.self_attn.v_proj.weight)
            self.slf_v_bias.append(mod.self_attn.v_proj.bias)
            self.slf_out_weight.append(mod.self_attn.out_proj.weight)
            self.slf_out_bias.append(mod.self_attn.out_proj.bias)

            self.cross_ln_weight.append(mod.norm2.weight)
            self.cross_ln_bias.append(mod.norm2.bias)
            self.cross_q_weight.append(mod.cross_attn.q_proj.weight)
            self.cross_q_bias.append(mod.cross_attn.q_proj.bias)
            self.cross_k_weight.append(mod.cross_attn.k_proj.weight)
            self.cross_k_bias.append(mod.cross_attn.k_proj.bias)
            self.cross_v_weight.append(mod.cross_attn.v_proj.weight)
            self.cross_v_bias.append(mod.cross_attn.v_proj.bias)
            self.cross_out_weight.append(mod.cross_attn.out_proj.weight)
            self.cross_out_bias.append(mod.cross_attn.out_proj.bias)

            self.ffn_ln_weight.append(mod.norm3.weight)
            self.ffn_ln_bias.append(mod.norm3.bias)
            self.ffn_inter_weight.append(mod.linear1.weight)
            self.ffn_inter_bias.append(mod.linear1.bias)
            self.ffn_out_weight.append(mod.linear2.weight)
            self.ffn_out_bias.append(mod.linear2.bias)

        self.decoder_ln_weight = [model.decoder.decoder.norm.weight]
        self.decoder_ln_bias = [model.decoder.decoder.norm.bias]

        self.mbart_ln_weight = [
            model.decoder.decoder_layernorm_embedding.weight
        ]
        self.mbart_ln_bias = [model.decoder.decoder_layernorm_embedding.bias]

        self.pos_emb = [model.decoder.decoder_embed_positions.weight]
        self.word_emb = [model.decoder.embed_tokens.weight]

        self.linear_weight = [model.lm_head_weight.t()]
        self.linear_bias = [model.final_logits_bias]

    def forward(self,
                enc_output,
                memory_seq_lens,
                trg_word=None,
                beam_size=4,
                top_k=1,
                top_p=0.0,
                decoding_strategy="beam_search_v3",
                max_out_len=256,
                diversity_rate=0.0,
                rel_len=False,
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
                alpha=0.6,
                temperature=1.0,
                early_stopping=False):
        # Beam_search/beam_search_v2/beam_search_v3 should be corrected to beam_search_v3.
        if decoding_strategy.startswith("beam_search"):
            decoding_strategy = "beam_search_v3"
        elif decoding_strategy == "greedy_search":
            decoding_strategy = "topk_sampling"
            top_k = 1
            top_p = 0.0
        elif decoding_strategy in [
                "sampling", "topk_sampling", "topp_sampling"
        ]:
            if top_p == 1 and top_k > 0:
                decoding_strategy = "topk_sampling"
                top_p = 0.0
            elif top_p > 0 and top_k == 0:
                decoding_strategy = "topp_sampling"
            else:
                raise AttributeError(
                    "Only topk sampling or topp sampling are supported. " \
                    "Topk sampling and topp sampling cannot be both applied in the faster version. ")
        output_ids, parent_ids, sequence_length = infer_mbart_decoding(
            [enc_output], [memory_seq_lens], self.word_emb, self.slf_ln_weight,
            self.slf_ln_bias, self.slf_q_weight, self.slf_q_bias,
            self.slf_k_weight, self.slf_k_bias, self.slf_v_weight,
            self.slf_v_bias, self.slf_out_weight, self.slf_out_bias,
            self.cross_ln_weight, self.cross_ln_bias, self.cross_q_weight,
            self.cross_q_bias, self.cross_k_weight, self.cross_k_bias,
            self.cross_v_weight, self.cross_v_bias, self.cross_out_weight,
            self.cross_out_bias, self.ffn_ln_weight, self.ffn_ln_bias,
            self.ffn_inter_weight, self.ffn_inter_bias, self.ffn_out_weight,
            self.ffn_out_bias, self.decoder_ln_weight, self.decoder_ln_bias,
            self.mbart_ln_weight, self.mbart_ln_bias, self.linear_weight,
            self.linear_bias, self.pos_emb, trg_word, decoding_strategy,
            beam_size, top_k, top_p, self._n_head,
            int(self._d_model / self._n_head), self._num_decoder_layers,
            bos_token_id, eos_token_id, max_out_len, -diversity_rate, rel_len,
            alpha, temperature, early_stopping, self._hidden_act)

        ids = finalize(
            beam_size,
            output_ids,
            parent_ids,
            sequence_length,
            decoding_strategy=decoding_strategy)
        return ids
