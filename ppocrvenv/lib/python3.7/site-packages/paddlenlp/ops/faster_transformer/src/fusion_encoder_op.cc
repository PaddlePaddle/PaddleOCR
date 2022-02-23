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

#include <string>
#include <vector>

#include "fusion_encoder_op.h"

std::vector<paddle::Tensor> EncoderForward(
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
    const int64_t& head_num,
    const int64_t& size_per_head,
    const bool& use_gelu,
    const bool& remove_padding,
    const int64_t& int8_mode,
    const int64_t& num_layer,
    const int64_t& layer_idx,
    const bool& allow_gemm_test,
    const bool& use_trt_kernel,
    const bool& normalize_before) {
  if (input.place() == paddle::PlaceType::kGPU) {
    auto shape = input.shape();
    auto encoder_out = paddle::Tensor(paddle::PlaceType::kGPU, shape);
    return EncoderCUDAForward(input,
                              attn_query_weight,
                              attn_query_bias,
                              attn_key_weight,
                              attn_key_bias,
                              attn_value_weight,
                              attn_value_bias,
                              attn_output_weight,
                              attn_output_bias,
                              attn_mask,
                              attn_output_layernorm_weight,
                              attn_output_layernorm_bias,
                              output_layernorm_weight,
                              output_layernorm_bias,
                              ffn_intermediate_weight,
                              ffn_intermediate_bias,
                              ffn_output_weight,
                              ffn_output_bias,
                              // sequence_id_offset,
                              // trt_seqlen_offset,
                              // amax_list,
                              encoder_out,
                              head_num,
                              size_per_head,
                              use_gelu,
                              remove_padding,
                              int8_mode,  // no support now
                              num_layer,
                              layer_idx,
                              allow_gemm_test,
                              use_trt_kernel,
                              normalize_before);
  } else {
    PD_THROW("Not implemented place. Only GPU is supported. ");
  }
}

std::vector<std::vector<int64_t>> EncoderInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& attn_query_weight_shape,
    const std::vector<int64_t>& attn_query_bias_shape,
    const std::vector<int64_t>& attn_key_weight_shape,
    const std::vector<int64_t>& attn_key_bias_shape,
    const std::vector<int64_t>& attn_value_weight_shape,
    const std::vector<int64_t>& attn_value_bias_shape,
    const std::vector<int64_t>& attn_output_weight_shape,
    const std::vector<int64_t>& attn_output_bias_shape,
    const std::vector<int64_t>& attn_mask_shape,
    const std::vector<int64_t>& attn_output_layernorm_weight_shape,
    const std::vector<int64_t>& attn_output_layernorm_bias_shape,
    const std::vector<int64_t>& output_layernorm_weight_shape,
    const std::vector<int64_t>& output_layernorm_bias_shape,
    const std::vector<int64_t>& ffn_intermediate_weight_shape,
    const std::vector<int64_t>& ffn_intermediate_bias_shape,
    const std::vector<int64_t>& ffn_output_weight_shape,
    const std::vector<int64_t>& ffn_output_bias_shape,
    // const std::vector<int64_t>& sequence_id_offset,
    // const std::vector<int64_t>& trt_seqlen_offset,
    // const std::vector<int64_t>& amax_list_shape,
    const int64_t& head_num,
    const int64_t& size_per_head,
    const bool& use_gelu,
    const bool& remove_padding,
    const int64_t& int8_mode,  // no support now
    const int64_t& num_layer,
    const int64_t& layer_idx,
    const bool& allow_gemm_test,
    const bool& use_trt_kernel,
    const bool& normalize_before) {
  return {input_shape};
}


std::vector<paddle::DataType> EncoderInferDtype(
    const paddle::DataType& input,
    const paddle::DataType& attn_query_weight,
    const paddle::DataType& attn_query_bias,
    const paddle::DataType& attn_key_weight,
    const paddle::DataType& attn_key_bias,
    const paddle::DataType& attn_value_weight,
    const paddle::DataType& attn_value_bias,
    const paddle::DataType& attn_output_weight,
    const paddle::DataType& attn_output_bias,
    const paddle::DataType& attn_mask,
    const paddle::DataType& attn_output_layernorm_weight,
    const paddle::DataType& attn_output_layernorm_bias,
    const paddle::DataType& output_layernorm_weight,
    const paddle::DataType& output_layernorm_bias,
    const paddle::DataType& ffn_intermediate_weight,
    const paddle::DataType& ffn_intermediate_bias,
    const paddle::DataType& ffn_output_weight,
    const paddle::DataType& ffn_output_bias) {
  // const paddle::DataType& sequence_id_offset,
  // const paddle::DataType& trt_seqlen_offset,
  // const paddle::DataType& amax_list) {
  return {input};
}

PD_BUILD_OP(fusion_encoder)
    .Inputs({
        "Input",
        "SelfQueryWeight",
        "SelfQueryBias",
        "SelfKeyWeight",
        "SelfKeyBias",
        "SelfValueWeight",
        "SelfValueBias",
        "SelfAttnOutputWeight",
        "SelfAttnOutputBias",
        "SelfAttnMask",
        "SelfAttnOutputLayernormWeight",
        "SelfAttnOutputLayernormBias",
        "OutputLayernormWeight",
        "OutputLayernormBias",
        "FFNInterWeight",
        "FFNInterBias",
        "FFNOutputWeight",
        "FFNOutputBias",
        // "SequenceIdOffset",
        // "TRTSeqLenOffset",
        // "AmaxList",
    })
    .Outputs({"EncoderOut"})
    .Attrs({"head_num: int64_t",
            "size_per_head: int64_t",
            "use_gelu: bool",
            "remove_padding: bool",
            "int8_mode: int64_t",
            "num_layer: int64_t",
            "layer_idx: int64_t",
            "allow_gemm_test: bool",
            "use_trt_kernel: bool",
            "normalize_before: bool"})
    .SetKernelFn(PD_KERNEL(EncoderForward))
    .SetInferShapeFn(PD_INFER_SHAPE(EncoderInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(EncoderInferDtype));