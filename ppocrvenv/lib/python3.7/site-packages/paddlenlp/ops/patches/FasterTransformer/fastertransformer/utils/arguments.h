/*
 * Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Decoder transformer
 **/

#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>
#include "fastertransformer/utils/common.h"
#include "fastertransformer/utils/common_structure.h"
#include "fastertransformer/utils/nccl_utils.h"

namespace fastertransformer {

template <typename T>
class DecodingInitParam : public AbstractParam {
public:
  /* weights for masked_multi_head_attention */
  const T *embedding_table = nullptr;
  const T *embedding_kernel = nullptr;
  const T *embedding_bias = nullptr;

  // Used for unilm.
  const T *trans_kernel = nullptr;
  const T *trans_bias = nullptr;

  const T *memory_tensor = nullptr;
  const int *type_id = nullptr;
  const int *memory_sequence_length = nullptr;

  // Used for force decoding.
  const int *trg_word = nullptr;
  const int *trg_length = nullptr;

  const T *position_encoding_table = nullptr;

  // segment table
  const T *type_table = nullptr;

  LayerNormWeight<T> layernorm;
  LayerNormWeight<T> lm_layernorm;
  LayerNormWeight<T> mbart_layernorm;

  const T *logits_mask = nullptr;

  int *output_ids = nullptr;
  int *parent_ids = nullptr;
  int *sequence_length = nullptr;
  cublasHandle_t cublas_handle;
  cublasLtHandle_t cublaslt_handle;
  cudaStream_t stream;

  // For GPT model
  int request_batch_size;
  int request_input_len;
  int request_output_len = 0;
  int max_input_len;
  int *d_start_ids;
  const int *d_start_lengths;
  const T *d_attn_mask;

  virtual ~DecodingInitParam() {}
};

struct TransformerArguments {
  size_t batch_size_;
  size_t seq_len_;
  size_t head_num_;
  size_t size_per_head_;
  size_t hidden_units_;
};

struct DecodingArguments : public TransformerArguments {
  int decoder_layers_;
  int vocab_size_;
  int start_id_;
  int end_id_;
  int vocab_size_padded_;
};

struct DecodingSamplingArguments : public DecodingArguments {
  int candidate_num_;
  float probability_threshold_;
  size_t cub_temp_storage_size_{0};
  bool normalization_before_{true};
  int pos_offset_{0};     // For BART position embedding
  bool pos_bias_{false};  // For Unified position embedding
  ActivationType act_{ActivationType::RELU};

  int memory_max_seq_len_{0};
  float temperature_{1.0};
  float repeat_penalty_{1.0};
  bool prefix_lm_{false};
  bool is_mbart_{false};
};

struct DecodingBeamsearchArguments : public DecodingArguments {
  int beam_width_;
  int temp_storage_size_;
  float beam_search_diversity_rate_;
  float alpha_;  // power number for length penalty in beam search v2
  bool normalization_before_{true};
  int pos_offset_{0};     // For BART position embedding
  bool pos_bias_{false};  // For Unified position embedding
  ActivationType act_{ActivationType::RELU};

  int memory_max_seq_len_{0};
  bool prefix_lm_{false};
  int finished_candidate_num_{-1};
  bool early_stopping_{false};
  bool is_mbart_{false};
};

struct GptArguments : public DecodingSamplingArguments {
  int **start_ids_;
  int start_len_;
  float temperature_{2.0};
  float len_penalty{1.0};
  float repetition_penalty_{1.0};
  int *vocab_mask{nullptr};
  int min_gpu_num_{1};
};

struct TransformerSamplingArguments : public DecodingSamplingArguments {
  int **start_ids_;
  int start_len_;
  float temperature_{1.0};
  float len_penalty{1.0};
  float repetition_penalty_{1.0};
  int *vocab_mask{nullptr};
  bool normalization_before_{true};
  bool pos_bias_{true};
  int unk_id_{-1};
  int mask_id_{-1};
  ActivationType act_{ActivationType::GELU};
};

struct TransformerBeamsearchArguments : public DecodingBeamsearchArguments {
  int start_len_;
  float temperature_{2.0};
  float len_penalty{1.0};
  float repetition_penalty_{2.0};
  bool normalization_before_{true};
  bool pos_bias_{true};
  int unk_id_{-1};
  int mask_id_{-1};
  ActivationType act_{ActivationType::GELU};
};

}  // namespace fastertransformer
