/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <string>
#include <vector>

#include "fastertransformer/open_decoder.h"
#include "fastertransformer/utils/common.h"

#ifdef PADDLE_ON_INFERENCE
#include "paddle/include/experimental/ext_all.h"
#else
#include "paddle/extension.h"
#endif


std::vector<paddle::Tensor> DecoderCUDAForward(
    const paddle::Tensor& from_tensor,
    const paddle::Tensor& memory_tensor,
    const paddle::Tensor& mem_seq_len,
    const paddle::Tensor& self_ln_weight,
    const paddle::Tensor& self_ln_bias,
    const paddle::Tensor& self_q_weight,
    const paddle::Tensor& self_q_bias,
    const paddle::Tensor& self_k_weight,
    const paddle::Tensor& self_k_bias,
    const paddle::Tensor& self_v_weight,
    const paddle::Tensor& self_v_bias,
    const paddle::Tensor& self_out_weight,
    const paddle::Tensor& self_out_bias,
    const paddle::Tensor& cross_ln_weight,
    const paddle::Tensor& cross_ln_bias,
    const paddle::Tensor& cross_q_weight,
    const paddle::Tensor& cross_q_bias,
    const paddle::Tensor& cross_k_weight,
    const paddle::Tensor& cross_k_bias,
    const paddle::Tensor& cross_v_weight,
    const paddle::Tensor& cross_v_bias,
    const paddle::Tensor& cross_out_weight,
    const paddle::Tensor& cross_out_bias,
    const paddle::Tensor& ffn_ln_weight,
    const paddle::Tensor& ffn_ln_bias,
    const paddle::Tensor& ffn_inter_weight,
    const paddle::Tensor& ffn_inter_bias,
    const paddle::Tensor& ffn_out_weight,
    const paddle::Tensor& ffn_out_bias,
    const paddle::Tensor& old_self_cache_key,
    const paddle::Tensor& old_self_cache_value,
    const paddle::Tensor& old_mem_cache,
    const int step,
    paddle::Tensor& decoder_output,
    paddle::Tensor& new_self_cache_key,
    paddle::Tensor& new_self_cache_value,
    paddle::Tensor& new_mem_cache,
    int n_head,
    int size_per_head,
    int memory_hidden_dim,
    bool is_fuse_qkv);
