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

#include "fusion_force_decoding_op.h"
#include "pd_traits.h"


std::vector<paddle::Tensor> DecodingForward(
    const paddle::Tensor& input,
    const paddle::Tensor& mem_seq_len,
    const paddle::Tensor& word_embedding,
    const std::vector<paddle::Tensor>& self_ln_weight,
    const std::vector<paddle::Tensor>& self_ln_bias,
    const std::vector<paddle::Tensor>& self_q_weight,
    const std::vector<paddle::Tensor>& self_q_bias,
    const std::vector<paddle::Tensor>& self_k_weight,
    const std::vector<paddle::Tensor>& self_k_bias,
    const std::vector<paddle::Tensor>& self_v_weight,
    const std::vector<paddle::Tensor>& self_v_bias,
    const std::vector<paddle::Tensor>& self_out_weight,
    const std::vector<paddle::Tensor>& self_out_bias,
    const std::vector<paddle::Tensor>& cross_ln_weight,
    const std::vector<paddle::Tensor>& cross_ln_bias,
    const std::vector<paddle::Tensor>& cross_q_weight,
    const std::vector<paddle::Tensor>& cross_q_bias,
    const std::vector<paddle::Tensor>& cross_k_weight,
    const std::vector<paddle::Tensor>& cross_k_bias,
    const std::vector<paddle::Tensor>& cross_v_weight,
    const std::vector<paddle::Tensor>& cross_v_bias,
    const std::vector<paddle::Tensor>& cross_out_weight,
    const std::vector<paddle::Tensor>& cross_out_bias,
    const std::vector<paddle::Tensor>& ffn_ln_weight,
    const std::vector<paddle::Tensor>& ffn_ln_bias,
    const std::vector<paddle::Tensor>& ffn_inter_weight,
    const std::vector<paddle::Tensor>& ffn_inter_bias,
    const std::vector<paddle::Tensor>& ffn_out_weight,
    const std::vector<paddle::Tensor>& ffn_out_bias,
    const paddle::Tensor& decoder_ln_weight,
    const paddle::Tensor& decoder_ln_bias,
    const paddle::Tensor& embedding_weight,
    const paddle::Tensor& embedding_bias,
    const paddle::Tensor& positional_embedding_weight,
    const paddle::Tensor& trg_word,
    const std::string& decoding_strategy,
    const int& beam_size,
    const int& topk,
    const float& topp,
    const int& n_head,
    const int& size_per_head,
    const int& num_layer,
    const int& bos_id,
    const int& eos_id,
    const int64_t& max_len,
    const float& beam_search_diversity_rate,
    const bool& rel_len,
    const float& alpha) {
  int batch_size = input.shape()[0];
  int max_out_len = rel_len ? max_len + input.shape()[1] : max_len;

  std::vector<int64_t> output_dims;
  std::vector<int64_t> parent_ids_dims;
  std::vector<int64_t> sequence_length_dims({batch_size});
  if (decoding_strategy == "beam_search") {
    batch_size /= beam_size;
    output_dims = {max_out_len, batch_size, beam_size};
    parent_ids_dims = output_dims;
  } else if (decoding_strategy == "beam_search_v2") {
    // Use separated alive and finish beam queues to avoid the decrease of alive
    // beams. The outputs must include both the finish and alive to trace full
    // path.
    if (batch_size != -1) {
      sequence_length_dims = {batch_size * 2};
      batch_size /= beam_size;
    } else {
      sequence_length_dims = {batch_size};
    }
    output_dims = {max_out_len, batch_size, beam_size * 2};
    parent_ids_dims = output_dims;
  } else if (decoding_strategy == "sampling") {
    output_dims = {max_out_len, batch_size};
    parent_ids_dims = {1};
  } else {
    PD_THROW("Not supported decoding strategy. ");
  }

  if (input.place() == paddle::PlaceType::kGPU) {
    auto output_ids = paddle::Tensor(paddle::PlaceType::kGPU, output_dims);
    auto parent_ids = paddle::Tensor(paddle::PlaceType::kGPU, parent_ids_dims);
    auto sequence_length =
        paddle::Tensor(paddle::PlaceType::kGPU, sequence_length_dims);

    paddle::Tensor seq_len = paddle::Tensor(paddle::PlaceType::kGPU);

    if (mem_seq_len.place() != paddle::PlaceType::kGPU) {
      seq_len = mem_seq_len.copy_to<int>(paddle::PlaceType::kGPU);
    } else {
      seq_len = mem_seq_len;
    }

    return DecodingCUDAForward(input,
                               seq_len,
                               word_embedding,
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
                               decoder_ln_weight,
                               decoder_ln_bias,
                               embedding_weight,
                               embedding_bias,
                               positional_embedding_weight,
                               trg_word,
                               output_ids,
                               parent_ids,
                               sequence_length,
                               decoding_strategy,
                               beam_size,
                               topk,
                               topp,
                               n_head,
                               size_per_head,
                               num_layer,
                               bos_id,
                               eos_id,
                               max_out_len,
                               beam_search_diversity_rate,
                               alpha);
  } else {
    PD_THROW("Not implemented place. Only GPU is supported. ");
  }
}

std::vector<std::vector<int64_t>> DecodingInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& mem_seq_len_shape,
    const std::vector<int64_t>& word_embedding_shape,
    const std::vector<std::vector<int64_t>>& self_ln_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_ln_bias_shapes,
    const std::vector<std::vector<int64_t>>& self_q_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_q_bias_shapes,
    const std::vector<std::vector<int64_t>>& self_k_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_k_bias_shapes,
    const std::vector<std::vector<int64_t>>& self_v_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_v_bias_shapes,
    const std::vector<std::vector<int64_t>>& self_out_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_out_bias_shapes,
    const std::vector<std::vector<int64_t>>& cross_ln_weight_shapes,
    const std::vector<std::vector<int64_t>>& cross_ln_bias_shapes,
    const std::vector<std::vector<int64_t>>& cross_q_weight_shapes,
    const std::vector<std::vector<int64_t>>& cross_q_bias_shapes,
    const std::vector<std::vector<int64_t>>& cross_k_weight_shapes,
    const std::vector<std::vector<int64_t>>& cross_k_bias_shapes,
    const std::vector<std::vector<int64_t>>& cross_v_weight_shapes,
    const std::vector<std::vector<int64_t>>& cross_v_bias_shapes,
    const std::vector<std::vector<int64_t>>& cross_out_weight_shapes,
    const std::vector<std::vector<int64_t>>& cross_out_bias_shapes,
    const std::vector<std::vector<int64_t>>& ffn_ln_weight_shapes,
    const std::vector<std::vector<int64_t>>& ffn_ln_bias_shapes,
    const std::vector<std::vector<int64_t>>& ffn_inter_weight_shapes,
    const std::vector<std::vector<int64_t>>& ffn_inter_bias_shapes,
    const std::vector<std::vector<int64_t>>& ffn_out_weight_shapes,
    const std::vector<std::vector<int64_t>>& ffn_out_bias_shapes,
    const std::vector<int64_t>& decoder_ln_weight_shape,
    const std::vector<int64_t>& decoder_ln_bias_shape,
    const std::vector<int64_t>& embedding_weight_shape,
    const std::vector<int64_t>& embedding_bias_shape,
    const std::vector<int64_t>& positional_embedding_weight_shape,
    const std::vector<int64_t>& trg_word_shape,
    const std::string& decoding_strategy,
    const int& beam_size,
    const int& topk,
    const float& topp,
    const int& n_head,
    const int& size_per_head,
    const int& num_layer,
    const int& bos_id,
    const int& eos_id,
    const int64_t& max_len,
    const float& beam_search_diversity_rate,
    const bool& rel_len,
    const float& alpha) {
  int batch_size = input_shape[0];

  std::vector<int64_t> output_dims;
  std::vector<int64_t> sequence_length_dims({batch_size});
  if (decoding_strategy == "beam_search") {
    if (batch_size != -1) {
      batch_size /= beam_size;
    }
    output_dims = {max_len, batch_size, beam_size};
    return {output_dims, output_dims, sequence_length_dims};
  } else if (decoding_strategy == "beam_search_v2") {
    // Use separated alive and finish beam queues to avoid the decrease of alive
    // beams. The outputs must include both the finish and alive to trace full
    // path.
    if (batch_size != -1) {
      sequence_length_dims = {batch_size * 2};
      batch_size /= beam_size;
    } else {
      sequence_length_dims = {batch_size};
    }
    output_dims = {max_len, batch_size, beam_size * 2};
    return {output_dims, output_dims, sequence_length_dims};
  } else if (decoding_strategy == "sampling") {
    output_dims = {max_len, batch_size};
    return {output_dims, {1}, sequence_length_dims};
  } else {
    PD_THROW("Not supported decoding strategy. ");
  }
}

std::vector<paddle::DataType> DecodingInferDtype(
    const paddle::DataType& input,
    const paddle::DataType& mem_seq_len,
    const paddle::DataType& word_embedding,
    const std::vector<paddle::DataType>& self_ln_weight,
    const std::vector<paddle::DataType>& self_ln_bias,
    const std::vector<paddle::DataType>& self_q_weight,
    const std::vector<paddle::DataType>& self_q_bias,
    const std::vector<paddle::DataType>& self_k_weight,
    const std::vector<paddle::DataType>& self_k_bias,
    const std::vector<paddle::DataType>& self_v_weight,
    const std::vector<paddle::DataType>& self_v_bias,
    const std::vector<paddle::DataType>& self_out_weight,
    const std::vector<paddle::DataType>& self_out_bias,
    const std::vector<paddle::DataType>& cross_ln_weight,
    const std::vector<paddle::DataType>& cross_ln_bias,
    const std::vector<paddle::DataType>& cross_q_weight,
    const std::vector<paddle::DataType>& cross_q_bias,
    const std::vector<paddle::DataType>& cross_k_weight,
    const std::vector<paddle::DataType>& cross_k_bias,
    const std::vector<paddle::DataType>& cross_v_weight,
    const std::vector<paddle::DataType>& cross_v_bias,
    const std::vector<paddle::DataType>& cross_out_weight,
    const std::vector<paddle::DataType>& cross_out_bias,
    const std::vector<paddle::DataType>& ffn_ln_weight,
    const std::vector<paddle::DataType>& ffn_ln_bias,
    const std::vector<paddle::DataType>& ffn_inter_weight,
    const std::vector<paddle::DataType>& ffn_inter_bias,
    const std::vector<paddle::DataType>& ffn_out_weight,
    const std::vector<paddle::DataType>& ffn_out_bias,
    const paddle::DataType& decoder_ln_weight,
    const paddle::DataType& decoder_ln_bias,
    const paddle::DataType& embedding_weight,
    const paddle::DataType& embedding_bias,
    const paddle::DataType& positional_embedding_weight,
    const paddle::DataType& trg_word) {
  return {paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32};
}

PD_BUILD_OP(fusion_force_decoding)
    .Inputs({"Input",
             "MemSeqLen",
             "WordEmbedding",
             paddle::Vec("SelfLayernormWeight"),
             paddle::Vec("SelfLayernormBias"),
             paddle::Vec("SelfQueryWeight"),
             paddle::Vec("SelfQueryBias"),
             paddle::Vec("SelfKeyWeight"),
             paddle::Vec("SelfKeyBias"),
             paddle::Vec("SelfValueWeight"),
             paddle::Vec("SelfValueBias"),
             paddle::Vec("SelfOutWeight"),
             paddle::Vec("SelfOutBias"),
             paddle::Vec("CrossLayernormWeight"),
             paddle::Vec("CrossLayernormBias"),
             paddle::Vec("CrossQueryWeight"),
             paddle::Vec("CrossQueryBias"),
             paddle::Vec("CrossKeyWeight"),
             paddle::Vec("CrossKeyBias"),
             paddle::Vec("CrossValueWeight"),
             paddle::Vec("CrossValueBias"),
             paddle::Vec("CrossOutWeight"),
             paddle::Vec("CrossOutBias"),
             paddle::Vec("FFNLayernormWeight"),
             paddle::Vec("FFNLayernormBias"),
             paddle::Vec("FFNInterWeight"),
             paddle::Vec("FFNInterBias"),
             paddle::Vec("FFNOutWeight"),
             paddle::Vec("FFNOutBias"),
             "DecoderLayernormWeight",
             "DecoderLayernormBias",
             "EmbWeight",
             "EmbBias",
             "PositionEncEmb",
             "TrgWord"})
    .Outputs({"OutputIds", "ParentIds", "SequenceLength"})
    .Attrs({"decoding_strategy: std::string",
            "beam_size: int",
            "topk: int",
            "topp: float",
            "n_head: int",
            "size_per_head: int",
            "num_layer: int",
            "bos_id: int",
            "eos_id: int",
            "max_len: int64_t",
            "beam_search_diversity_rate: float",
            "rel_len: bool",
            "alpha: float"})
    .SetKernelFn(PD_KERNEL(DecodingForward))
    .SetInferShapeFn(PD_INFER_SHAPE(DecodingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DecodingInferDtype));
