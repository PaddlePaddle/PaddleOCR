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

#include "fastertransformer/bert_encoder_transformer.h"
#include "fastertransformer/utils/common.h"

#ifdef PADDLE_ON_INFERENCE
#include "paddle/include/experimental/ext_all.h"
#else
#include "paddle/extension.h"
#endif


std::vector<paddle::Tensor> EncoderCUDAForward(
    const paddle::Tensor& input,
    const paddle::Tensor& attn_query_weight,
    const paddle::Tensor& attn_query_bias,
    const paddle::Tensor& attn_key_weight,
    const paddle::Tensor& attn_key_bias,
    const paddle::Tensor& attn_value_weight,
    const paddle::Tensor& attn_value_bias,
    const paddle::Tensor& attn_output_weight,
    const paddle::Tensor& attn_output_bias,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& attn_output_layernorm_weight,
    const paddle::Tensor& attn_output_layernorm_bias,
    const paddle::Tensor& output_layernorm_weight,
    const paddle::Tensor& output_layernorm_bias,
    const paddle::Tensor& ffn_intermediate_weight,
    const paddle::Tensor& ffn_intermediate_bias,
    const paddle::Tensor& ffn_output_weight,
    const paddle::Tensor& ffn_output_bias,
    // const paddle::Tensor& sequence_id_offset,
    // const paddle::Tensor& trt_seqlen_offset,
    // const paddle::Tensor& amax_list,
    paddle::Tensor& encoder_out,
    int64_t head_num_,
    int64_t size_per_head_,
    bool use_gelu,
    bool remove_padding,
    int64_t int8_mode,  // no support now
    int64_t num_layer_,
    int64_t layer_idx_,
    bool allow_gemm_test,
    bool use_trt_kernel_,
    bool normalize_before);