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

#include "fusion_decoder_op.h"
#include "pd_traits.h"


std::vector<paddle::Tensor> DecoderForward(
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
    int n_head,
    int size_per_head,
    int memory_hidden_dim,
    bool is_fuse_qkv) {
  const int batch_size = memory_tensor.shape()[0];
  std::vector<int64_t> output_dims;
  output_dims = {batch_size, 1, n_head * size_per_head};

  auto new_self_cache_key = old_self_cache_key;
  auto new_self_cache_value = old_self_cache_value;
  auto new_mem_cache = old_mem_cache;

  if (from_tensor.place() == paddle::PlaceType::kGPU) {
    auto decoder_output = paddle::Tensor(from_tensor.place(), output_dims);

    paddle::Tensor _mem_seq_len = paddle::Tensor(paddle::PlaceType::kGPU);

    if (mem_seq_len.place() != paddle::PlaceType::kGPU) {
      _mem_seq_len = mem_seq_len.copy_to<int>(paddle::PlaceType::kGPU);
    } else {
      _mem_seq_len = mem_seq_len;
    }

    return DecoderCUDAForward(from_tensor,
                              memory_tensor,
                              _mem_seq_len,
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
                              decoder_output,
                              new_self_cache_key,
                              new_self_cache_value,
                              new_mem_cache,
                              n_head,
                              size_per_head,
                              memory_hidden_dim,
                              is_fuse_qkv);
  } else {
    PD_THROW("Not implemented place. Only GPU is supported. ");
  }
}

std::vector<std::vector<int64_t>> DecoderInferShape(
    const std::vector<int64_t>& from_tensor_shape,
    const std::vector<int64_t>& memory_tensor_shape,
    const std::vector<int64_t>& mem_seq_len_shape,
    const std::vector<int64_t>& self_ln_weight_shapes,
    const std::vector<int64_t>& self_ln_bias_shapes,
    const std::vector<int64_t>& self_q_weight_shapes,
    const std::vector<int64_t>& self_q_bias_shapes,
    const std::vector<int64_t>& self_k_weight_shapes,
    const std::vector<int64_t>& self_k_bias_shapes,
    const std::vector<int64_t>& self_v_weight_shapes,
    const std::vector<int64_t>& self_v_bias_shapes,
    const std::vector<int64_t>& self_out_weight_shapes,
    const std::vector<int64_t>& self_out_bias_shapes,
    const std::vector<int64_t>& cross_ln_weight_shapes,
    const std::vector<int64_t>& cross_ln_bias_shapes,
    const std::vector<int64_t>& cross_q_weight_shapes,
    const std::vector<int64_t>& cross_q_bias_shapes,
    const std::vector<int64_t>& cross_k_weight_shapes,
    const std::vector<int64_t>& cross_k_bias_shapes,
    const std::vector<int64_t>& cross_v_weight_shapes,
    const std::vector<int64_t>& cross_v_bias_shapes,
    const std::vector<int64_t>& cross_out_weight_shapes,
    const std::vector<int64_t>& cross_out_bias_shapes,
    const std::vector<int64_t>& ffn_ln_weight_shapes,
    const std::vector<int64_t>& ffn_ln_bias_shapes,
    const std::vector<int64_t>& ffn_inter_weight_shapes,
    const std::vector<int64_t>& ffn_inter_bias_shapes,
    const std::vector<int64_t>& ffn_out_weight_shapes,
    const std::vector<int64_t>& ffn_out_bias_shapes,
    const std::vector<int64_t>& old_self_cache_key_shape,
    const std::vector<int64_t>& old_self_cache_value_shape,
    const std::vector<int64_t>& old_mem_cache_shape,
    const int& step,
    const int& n_head,
    const int& size_per_head,
    const int& memory_hidden_dim,
    const bool& is_fuse_qkv) {
  return {from_tensor_shape,
          old_self_cache_key_shape,
          old_self_cache_value_shape,
          old_mem_cache_shape};
}

std::vector<paddle::DataType> DecoderInferDtype(
    const paddle::DataType& from_tensor,
    const paddle::DataType& memory_tensor,
    const paddle::DataType& mem_seq_len,
    const paddle::DataType& self_ln_weight,
    const paddle::DataType& self_ln_bias,
    const paddle::DataType& self_q_weight,
    const paddle::DataType& self_q_bias,
    const paddle::DataType& self_k_weight,
    const paddle::DataType& self_k_bias,
    const paddle::DataType& self_v_weight,
    const paddle::DataType& self_v_bias,
    const paddle::DataType& self_out_weight,
    const paddle::DataType& self_out_bias,
    const paddle::DataType& cross_ln_weight,
    const paddle::DataType& cross_ln_bias,
    const paddle::DataType& cross_q_weight,
    const paddle::DataType& cross_q_bias,
    const paddle::DataType& cross_k_weight,
    const paddle::DataType& cross_k_bias,
    const paddle::DataType& cross_v_weight,
    const paddle::DataType& cross_v_bias,
    const paddle::DataType& cross_out_weight,
    const paddle::DataType& cross_out_bias,
    const paddle::DataType& ffn_ln_weight,
    const paddle::DataType& ffn_ln_bias,
    const paddle::DataType& ffn_inter_weight,
    const paddle::DataType& ffn_inter_bias,
    const paddle::DataType& ffn_out_weight,
    const paddle::DataType& ffn_out_bias,
    const paddle::DataType& old_self_cache_key,
    const paddle::DataType& old_self_cache_value,
    const paddle::DataType& old_mem_cache) {
  return {from_tensor, old_self_cache_key, old_self_cache_value, old_mem_cache};
}

PD_BUILD_OP(fusion_decoder)
    .Inputs(
        {"FromTensor",          "MemoryTensor",         "MemSeqLen",
         "SelfLayernormWeight", "SelfLayernormBias",    "SelfQueryWeight",
         "SelfQueryBias",       "SelfKeyWeight",        "SelfKeyBias",
         "SelfValueWeight",     "SelfValueBias",        "SelfOutWeight",
         "SelfOutBias",         "CrossLayernormWeight", "CrossLayernormBias",
         "CrossQueryWeight",    "CrossQueryBias",       "CrossKeyWeight",
         "CrossKeyBias",        "CrossValueWeight",     "CrossValueBias",
         "CrossOutWeight",      "CrossOutBias",         "FFNLayernormWeight",
         "FFNLayernormBias",    "FFNInterWeight",       "FFNInterBias",
         "FFNOutWeight",        "FFNOutBias",           "OldSelfCacheKey",
         "OldSelfCacheValue",   "OldMemCache"})
    .Outputs({"DecoderOutput",
              "NewSelfCacheKey",
              "NewSelfCacheValue",
              "NewMemCache"})
    .Attrs({"step: int",
            "n_head: int",
            "size_per_head: int",
            "memory_hidden_dim: int",
            "is_fuse_qkv: bool"})
    .SetKernelFn(PD_KERNEL(DecoderForward))
    .SetInferShapeFn(PD_INFER_SHAPE(DecoderInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DecoderInferDtype));
