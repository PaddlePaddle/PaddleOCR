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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.fluid.layer_helper import LayerHelper
from paddlenlp.transformers import WordEmbedding, PositionalEmbedding, position_encoding_init
from paddlenlp.utils.log import logger
from paddlenlp.ops.ext_utils import load, LOADED_EXT
from paddlenlp.ops import transfer_param


def infer_transformer_decoder(from_tensor,
                              memory_tensor,
                              mem_seq_len,
                              self_ln_weight,
                              self_ln_bias,
                              self_q_weight,
                              self_q_bias,
                              self_k_weight,
                              self_k_bias,
                              self_v_weight,
                              self_v_bias,
                              self_out_weight,
                              self_out_bias,
                              cross_ln_weight,
                              cross_ln_bias,
                              cross_q_weight,
                              cross_q_bias,
                              cross_k_weight,
                              cross_k_bias,
                              cross_v_weight,
                              cross_v_bias,
                              cross_out_weight,
                              cross_out_bias,
                              ffn_ln_weight,
                              ffn_ln_bias,
                              ffn_inter_weight,
                              ffn_inter_bias,
                              ffn_out_weight,
                              ffn_out_bias,
                              old_self_cache_key,
                              old_self_cache_value,
                              old_mem_cache,
                              step,
                              n_head,
                              size_per_head,
                              memory_hidden_dim,
                              is_fuse_qkv=False):
    helper = LayerHelper('fusion_decoder', **locals())

    inputs = {
        "FromTensor": from_tensor,
        "MemoryTensor": memory_tensor,
        "MemSeqLen": mem_seq_len,
        "SelfLayernormWeight": self_ln_weight,
        "SelfLayernormBias": self_ln_bias,
        "SelfQueryWeight": self_q_weight,
        "SelfQueryBias": self_q_bias,
        "SelfKeyWeight": self_k_weight,
        "SelfKeyBias": self_k_bias,
        "SelfValueWeight": self_v_weight,
        "SelfValueBias": self_v_bias,
        "SelfOutWeight": self_out_weight,
        "SelfOutBias": self_out_bias,
        "CrossLayernormWeight": cross_ln_weight,
        "CrossLayernormBias": cross_ln_bias,
        "CrossQueryWeight": cross_q_weight,
        "CrossQueryBias": cross_q_bias,
        "CrossKeyWeight": cross_k_weight,
        "CrossKeyBias": cross_k_bias,
        "CrossValueWeight": cross_v_weight,
        "CrossValueBias": cross_v_bias,
        "CrossOutWeight": cross_out_weight,
        "CrossOutBias": cross_out_bias,
        "FFNLayernormWeight": ffn_ln_weight,
        "FFNLayernormBias": ffn_ln_bias,
        "FFNInterWeight": ffn_inter_weight,
        "FFNInterBias": ffn_inter_bias,
        "FFNOutWeight": ffn_out_weight,
        "FFNOutBias": ffn_out_bias,
        "OldSelfCacheKey": old_self_cache_key,
        "OldSelfCacheValue": old_self_cache_value,
        "OldMemCache": old_mem_cache
    }
    attrs = {
        'step': step,
        'n_head': n_head,
        'size_per_head': size_per_head,
        'memory_hidden_dim': memory_hidden_dim,
        'is_fuse_qkv': is_fuse_qkv
    }

    decoder_output = helper.create_variable(dtype=memory_tensor.dtype)
    new_self_cache_key = helper.create_variable(dtype=memory_tensor.dtype)
    new_self_cache_value = helper.create_variable(dtype=memory_tensor.dtype)
    new_mem_cache = helper.create_variable(dtype=memory_tensor.dtype)

    outputs = {
        'DecoderOutput': decoder_output,
        'NewSelfCacheKey': new_self_cache_key,
        'NewSelfCacheValue': new_self_cache_value,
        'NewMemCache': new_mem_cache
    }

    helper.append_op(
        type='fusion_decoder', inputs=inputs, outputs=outputs, attrs=attrs)

    return decoder_output, new_self_cache_key, new_self_cache_value, new_mem_cache


def get_op_cache_config(use_batch_major_op_cache, size_per_head, is_fp16):
    x = 8 if is_fp16 else 4
    use_batch_major_op_cache = True if use_batch_major_op_cache == True and \
                                       size_per_head % x == 0 \
                                    else False
    x = x if use_batch_major_op_cache else 1
    return use_batch_major_op_cache, x


class InferTransformerDecoder(nn.Layer):
    """
    FasterTransformer decoder block.

    Args:
        decoder (`TransformerDecoder`):
            Transformer decoder block.
        n_head (`int`):
            The number of head used in multi-head attention.
        size_per_head (`int`):
            The size of per head used in multi-head attention.
        decoder_lib (`str`):
            The path to decoder_lib. Default to None.
        use_fp16_decoder (`bool`):
            Whether to use fp16 for decoder. Default to False.
    """

    def __init__(self,
                 decoder,
                 n_head,
                 size_per_head,
                 decoder_lib=None,
                 use_fp16_decoder=False,
                 use_batch_major_op_cache=False):

        if decoder_lib is not None and os.path.isfile(decoder_lib):
            # Maybe it has been loadad by `ext_utils.load`
            if "FasterTransformer" not in LOADED_EXT.keys():
                ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                    decoder_lib)
                LOADED_EXT["FasterTransformer"] = ops
        else:
            if decoder_lib is not None:
                logger.warning(
                    "The specified decoder_lib does not exist, and it will be built automatically."
                )
            load("FasterTransformer", verbose=True)

        super(InferTransformerDecoder, self).__init__()
        self.n_head = n_head
        self.size_per_head = size_per_head
        self.use_batch_major_op_cache = use_batch_major_op_cache

        if use_fp16_decoder:
            for idx, mod in enumerate(decoder.layers):
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

        self.weights = []
        for idx, mod in enumerate(decoder.layers):
            layer_weight = []
            layer_weight.append(mod.norm1.weight)
            layer_weight.append(mod.norm1.bias)
            layer_weight.append(mod.self_attn.q_proj.weight)
            layer_weight.append(mod.self_attn.q_proj.bias)
            layer_weight.append(mod.self_attn.k_proj.weight)
            layer_weight.append(mod.self_attn.k_proj.bias)
            layer_weight.append(mod.self_attn.v_proj.weight)
            layer_weight.append(mod.self_attn.v_proj.bias)
            layer_weight.append(mod.self_attn.out_proj.weight)
            layer_weight.append(mod.self_attn.out_proj.bias)
            layer_weight.append(mod.norm2.weight)
            layer_weight.append(mod.norm2.bias)
            layer_weight.append(mod.cross_attn.q_proj.weight)
            layer_weight.append(mod.cross_attn.q_proj.bias)
            layer_weight.append(mod.cross_attn.k_proj.weight)
            layer_weight.append(mod.cross_attn.k_proj.bias)
            layer_weight.append(mod.cross_attn.v_proj.weight)
            layer_weight.append(mod.cross_attn.v_proj.bias)
            layer_weight.append(mod.cross_attn.out_proj.weight)
            layer_weight.append(mod.cross_attn.out_proj.bias)
            layer_weight.append(mod.norm3.weight)
            layer_weight.append(mod.norm3.bias)
            layer_weight.append(mod.linear1.weight)
            layer_weight.append(mod.linear1.bias)
            layer_weight.append(mod.linear2.weight)
            layer_weight.append(mod.linear2.bias)
            self.weights.append(layer_weight)

    def forward(self, from_tensor, memory_tensor, mem_seq_len, self_cache_key,
                self_cache_value, mem_cache, step, memory_hidden_dim,
                is_fuse_qkv):
        decoder_output = from_tensor
        self_caches_key = []
        self_caches_value = []
        mem_caches = []
        if not self.use_batch_major_op_cache:
            self_cache_key = paddle.concat(
                [
                    self_cache_key, paddle.zeros(
                        shape=[
                            len(self.weights), 1,
                            paddle.shape(memory_tensor)[0],
                            self.n_head * self.size_per_head
                        ],
                        dtype=self_cache_key.dtype)
                ],
                axis=1)
            self_cache_value = paddle.concat(
                [
                    self_cache_value, paddle.zeros(
                        shape=[
                            len(self.weights), 1,
                            paddle.shape(memory_tensor)[0],
                            self.n_head * self.size_per_head
                        ],
                        dtype=self_cache_value.dtype)
                ],
                axis=1)
        for idx in range(len(self.weights)):
            weight = self.weights[idx]
            decoder_output, new_self_cache_key, new_self_cache_value, new_mem_cache = infer_transformer_decoder(
                from_tensor=decoder_output,
                memory_tensor=memory_tensor,
                mem_seq_len=mem_seq_len,
                self_ln_weight=weight[0],
                self_ln_bias=weight[1],
                self_q_weight=weight[2],
                self_q_bias=weight[3],
                self_k_weight=weight[4],
                self_k_bias=weight[5],
                self_v_weight=weight[6],
                self_v_bias=weight[7],
                self_out_weight=weight[8],
                self_out_bias=weight[9],
                cross_ln_weight=weight[10],
                cross_ln_bias=weight[11],
                cross_q_weight=weight[12],
                cross_q_bias=weight[13],
                cross_k_weight=weight[14],
                cross_k_bias=weight[15],
                cross_v_weight=weight[16],
                cross_v_bias=weight[17],
                cross_out_weight=weight[18],
                cross_out_bias=weight[19],
                ffn_ln_weight=weight[20],
                ffn_ln_bias=weight[21],
                ffn_inter_weight=weight[22],
                ffn_inter_bias=weight[23],
                ffn_out_weight=weight[24],
                ffn_out_bias=weight[25],
                old_self_cache_key=self_cache_key[idx],
                old_self_cache_value=self_cache_value[idx],
                old_mem_cache=mem_cache[idx],
                step=step,
                n_head=self.n_head,
                size_per_head=self.size_per_head,
                memory_hidden_dim=memory_hidden_dim,
                is_fuse_qkv=is_fuse_qkv)
            self_caches_key.append(new_self_cache_key)
            self_caches_value.append(new_self_cache_value)
            mem_caches.append(new_mem_cache)

        self_cache_key = paddle.stack(self_caches_key, axis=0)
        self_cache_value = paddle.stack(self_caches_value, axis=0)
        mem_cache = paddle.stack(mem_caches, axis=0)
        return decoder_output, self_cache_key, self_cache_value, mem_cache


class FasterDecoder(nn.Layer):
    """
    FasterTransformer decoder for auto-regressive generation.

    Args:
        src_vocab_size (`int`):
            The size of source vocabulary.
        trg_vocab_size (`int`):
            The size of target vocabulary.
        max_length (`int`):
            The maximum length of input sequences.
        num_encoder_layers (`int`):
            The number of sub-layers to be stacked in the encoder.
        num_decoder_layers (`int`):
            The number of sub-layers to be stacked in the decoder.
        n_head (`int`):
            The number of head used in multi-head attention.
        d_model (`int`):
            The dimension for word embeddings, which is also the last dimension of
            the input and output of multi-head attention, position-wise feed-forward
            networks, encoder and decoder.
        d_inner_hid (`int`):
            Size of the hidden layer in position-wise feed-forward networks.
        dropout (`float`):
            Dropout rates. Used for pre-process, activation and inside attention.
        weight_sharing (`bool`):
            Whether to use weight sharing.
        bos_id (`int`, optional):
            The start token id and also is used as padding id. Defaults to 0.
        eos_id (`int`, optional):
            The end token id. Defaults to 1.
        max_out_len (int, optional):
            The maximum output length. Defaults to 256.
        decoder_lib (`str`):
            The path to decoder_lib. Default to None.
        use_fp16_decoder (`bool`):
            Whether to use fp16 for decoder. Default to False.
    """

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 max_length,
                 num_encoder_layers,
                 num_decoder_layers,
                 n_head,
                 d_model,
                 d_inner_hid,
                 dropout,
                 weight_sharing,
                 bos_id=0,
                 eos_id=1,
                 max_out_len=256,
                 decoder_lib=None,
                 use_fp16_decoder=False,
                 use_batch_major_op_cache=False):
        super().__init__()
        self.trg_vocab_size = trg_vocab_size
        self.n_head = n_head
        self.emb_dim = d_model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.dropout = dropout
        self.max_out_len = max_out_len
        self.max_length = max_length
        self.use_fp16_decoder = use_fp16_decoder
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.size_per_head = d_model // n_head
        self.use_batch_major_op_cache, self.x = get_op_cache_config(
            use_batch_major_op_cache, self.size_per_head, use_fp16_decoder)

        self.src_word_embedding = WordEmbedding(
            vocab_size=src_vocab_size, emb_dim=d_model, bos_id=self.bos_id)
        # print(self.src_word_embedding.word_embedding.weight)
        self.src_pos_embedding = PositionalEmbedding(
            emb_dim=d_model, max_length=max_length)
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )
            self.trg_word_embedding = self.src_word_embedding
            self.trg_pos_embedding = self.src_pos_embedding
        else:
            self.trg_word_embedding = WordEmbedding(
                vocab_size=trg_vocab_size, emb_dim=d_model, bos_id=self.bos_id)
            self.trg_pos_embedding = PositionalEmbedding(
                emb_dim=d_model, max_length=max_length)

        self.transformer = paddle.nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_inner_hid,
            dropout=dropout,
            activation="relu",
            normalize_before=True)

        self.decoder = InferTransformerDecoder(
            decoder=self.transformer.decoder,
            n_head=n_head,
            size_per_head=self.size_per_head,
            decoder_lib=decoder_lib,
            use_fp16_decoder=use_fp16_decoder,
            use_batch_major_op_cache=self.use_batch_major_op_cache)

        if weight_sharing:
            self.linear = lambda x: paddle.matmul(x=x,
                                                  y=self.trg_word_embedding.word_embedding.weight,
                                                  transpose_y=True)
        else:
            self.linear = nn.Linear(
                in_features=d_model,
                out_features=trg_vocab_size,
                bias_attr=False)

    def forward(self, src_word):
        src_max_len = paddle.shape(src_word)[-1]
        mem_seq_lens = paddle.sum(paddle.cast(
            src_word != self.bos_id, dtype="int32"),
                                  axis=-1,
                                  keepdim=True,
                                  dtype="int32")

        src_slf_attn_bias = paddle.cast(
            src_word == self.bos_id,
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e9

        src_slf_attn_bias.stop_gradient = True

        src_pos = paddle.cast(
            src_word != self.bos_id, dtype="int64") * paddle.arange(
                start=0, end=src_max_len)

        src_emb = self.src_word_embedding(src_word)

        src_pos_emb = self.src_pos_embedding(src_pos)
        src_emb = src_emb + src_pos_emb
        enc_input = F.dropout(
            src_emb, p=self.dropout,
            training=self.training) if self.dropout else src_emb
        enc_output = self.transformer.encoder(
            enc_input, src_mask=src_slf_attn_bias)

        batch_size, _, memory_hidden_dim = enc_output.shape
        end_token_tensor = paddle.full(
            shape=[batch_size, 1], fill_value=self.eos_id, dtype="int64")

        predict_ids = []
        log_probs = paddle.full(
            shape=[batch_size, 1], fill_value=0, dtype="float32")
        trg_word = paddle.full(
            shape=[batch_size, 1], fill_value=self.bos_id, dtype="int64")

        if self.use_fp16_decoder:
            enc_output = paddle.cast(enc_output, "float16")

        # Init cache
        if not self.use_batch_major_op_cache:
            self_cache_key = paddle.zeros(
                shape=[self.num_decoder_layers, 0, batch_size, self.d_model],
                dtype=enc_output.dtype)
            self_cache_value = paddle.zeros(
                shape=[self.num_decoder_layers, 0, batch_size, self.d_model],
                dtype=enc_output.dtype)
        else:
            self_cache_key = paddle.zeros(
                shape=[
                    self.num_decoder_layers, batch_size, self.n_head,
                    self.size_per_head // self.x, self.max_out_len, self.x
                ],
                dtype=enc_output.dtype)
            self_cache_value = paddle.zeros(
                shape=[
                    self.num_decoder_layers, batch_size, self.n_head,
                    self.max_out_len, self.size_per_head
                ],
                dtype=enc_output.dtype)
        mem_cache = paddle.zeros(
            shape=[
                self.num_decoder_layers, 2, batch_size, src_max_len,
                self.d_model
            ],
            dtype=enc_output.dtype)
        for i in range(self.max_out_len):
            trg_pos = paddle.full(
                shape=trg_word.shape, fill_value=i, dtype="int64")
            trg_emb = self.trg_word_embedding(trg_word)
            trg_pos_emb = self.trg_pos_embedding(trg_pos)
            trg_emb = trg_emb + trg_pos_emb
            dec_input = F.dropout(
                trg_emb, p=self.dropout,
                training=self.training) if self.dropout else trg_emb

            # TODO(gongenlei): do cast in op
            if self.use_fp16_decoder:
                dec_input = paddle.cast(dec_input, "float16")
            dec_output, self_cache_key, self_cache_value, mem_cache = self.decoder(
                from_tensor=dec_input,
                memory_tensor=enc_output,
                mem_seq_len=mem_seq_lens,
                self_cache_key=self_cache_key,
                self_cache_value=self_cache_value,
                mem_cache=mem_cache,
                step=i,
                memory_hidden_dim=memory_hidden_dim,
                is_fuse_qkv=False)

            if self.use_fp16_decoder:
                dec_output = paddle.cast(dec_output, "float32")

            dec_output = paddle.reshape(
                dec_output, shape=[-1, dec_output.shape[-1]])

            logits = self.linear(dec_output)
            step_log_probs = paddle.log(F.softmax(logits, axis=-1))
            log_probs = paddle.add(x=step_log_probs, y=log_probs)
            scores = log_probs
            topk_scores, topk_indices = paddle.topk(x=scores, k=1)

            finished = paddle.equal(topk_indices, end_token_tensor)
            trg_word = topk_indices
            log_probs = topk_scores

            predict_ids.append(topk_indices)

            # TODO(gongenlei): support static graph
            if paddle.all(finished).numpy():
                break

        predict_ids = paddle.stack(predict_ids, axis=0)
        finished_seq = paddle.transpose(predict_ids, [1, 2, 0])
        finished_scores = topk_scores

        return finished_seq, finished_scores

    def load(self, init_from_params):
        # Load the trained model
        assert init_from_params, (
            "Please set init_from_params to load the infer model.")

        model_dict = paddle.load(init_from_params, return_numpy=True)

        # To set weight[padding_idx] to 0.
        model_dict["trg_word_embedding.word_embedding.weight"][
            self.bos_id] = [0] * self.d_model

        # To avoid a longer length than training, reset the size of position
        # encoding to max_length
        model_dict["encoder.pos_encoder.weight"] = position_encoding_init(
            self.max_length, self.d_model)
        model_dict["decoder.pos_encoder.weight"] = position_encoding_init(
            self.max_length, self.d_model)

        if self.use_fp16_decoder:
            for item in self.state_dict():
                if "decoder.layers" in item:
                    model_dict[item] = np.float16(model_dict[item])

        self.load_dict(model_dict)
