/*
 * Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * Decoder for a Single Step of a Single Layer
 **/

#pragma once
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "fastertransformer/cuda/attention_kernels.cuh"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/cuda/open_decoder.cuh"
#include "fastertransformer/cuda/transformer_kernels.cuh"
#include "fastertransformer/utils/allocator.h"
#include "fastertransformer/utils/common.h"
#include "fastertransformer/utils/common_structure.h"
#include "fastertransformer/utils/functions.h"
#include "fastertransformer/utils/nccl_utils.h"
#include "fastertransformer/utils/nvtx_utils.h"
#include "fastertransformer/utils/nvtx_utils.h"

// use new attention implementation with [B, H, Dh/x, L, x] cache format for the
// keys
// and [B, H, L, Dh] for values

#define USE_CACHE_BATCH_MAJOR_ATTENTION 1

namespace fastertransformer {

template <typename T>
class DecoderInitParam : public AbstractParam {
public:
  /* weights for masked_multi_head_attention */
  LayerNormWeight<T> self_layernorm;
  AttentionWeight<T> self_attention;

  LayerNormWeight<T> cross_layernorm;
  AttentionWeight<T> cross_attention;

  LayerNormWeight<T> ffn_layernorm;
  FFNWeight<T> ffn;
  cublasHandle_t cublas_handle;
  cublasLtHandle_t cublaslt_handle;
  cudaStream_t stream;

  int request_batch_size = -1;
  int request_max_mem_seq_len = -1;

  const float *k_cache = nullptr;
  const float *v_cache = nullptr;
};

template <OperationType OpType_>
class DecoderTransformerTraits;

template <>
class DecoderTransformerTraits<OperationType::FP32>
    : public TransformerTraits<OperationType::FP32> {};

template <>
class DecoderTransformerTraits<OperationType::FP16>
    : public TransformerTraits<OperationType::FP16> {};

template <OperationType OpType_>
class OpenDecoder {
private:
  typedef DecoderTransformerTraits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  DecoderInitParam<DataType_> param_;
  TensorParallelParam t_parallel_param_;
  LayerParallelParam l_parallel_param_;

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  std::map<std::string, cublasLtMatmulAlgo_info> cublasAlgoMap_;

  int max_batch_size_ = -1;
  int head_num_;
  int size_per_head_;
  int hidden_units_;
  int memory_hidden_units_;
  int normalization_before_;
  ActivationType act_;

  DataType_ *norm_from_tensor_buf_, *query_buf_;
  DataType_ *context_buf_, *masked_output_buf_;
  DataType_ *norm_masked_output_buf_, *cross_output_buf_;
  DataType_ *norm_cross_output_buf_, *ffn_inner_buf_, *ffn_out_buf_;
  DataType_ *key_buf_, *value_buf_;

  DataType_ **qkv_kernel_;
  DataType_ **qkv_input_;
  DataType_ **qkv_buf_;
  void *cublas_workspace_ = nullptr;

  bool is_fuse_QKV_in_batched_gemm_;
  const bool is_fuse_QKV_in_normal_gemm_;

public:
  void judgeFusedQKV() {
    is_fuse_QKV_in_batched_gemm_ = false;
    int m, n, k, dataType;
    if (std::is_same<half, DataType_>::value)
      dataType = HALF_DATATYPE;
    else
      dataType = FLOAT_DATATYPE;

    m = l_parallel_param_.local_batch_size;
    n = t_parallel_param_.local_hidden_units_;
    k = hidden_units_;
    char mark[256], mark2[256];
    sprintf(mark, "1_%d_%d_%d_%d", n, m, k, dataType);
    sprintf(mark2, "3_%d_%d_%d_%d", n, m, k, dataType);
    if (cublasAlgoMap_.find(mark) != cublasAlgoMap_.end() &&
        cublasAlgoMap_.find(mark2) != cublasAlgoMap_.end() &&
        3 * cublasAlgoMap_[mark].exec_time > cublasAlgoMap_[mark2].exec_time) {
      is_fuse_QKV_in_batched_gemm_ = true;
    }
  }


  OpenDecoder(int head_num,
              int size_per_head,
              int memory_hidden_units,
              bool is_fuse_QKV_in_normal_gemm = false,
              bool normalization_before = true,
              ActivationType act = ActivationType::GELU)
      // Activation function default to GELU for GPT.
      : head_num_(head_num),
        size_per_head_(size_per_head),
        memory_hidden_units_(memory_hidden_units),
        is_fuse_QKV_in_normal_gemm_(is_fuse_QKV_in_normal_gemm),
        normalization_before_(normalization_before),
        act_(act) {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    hidden_units_ = head_num_ * size_per_head_;
    t_parallel_param_.local_head_num_ = head_num_;
    t_parallel_param_.local_hidden_units_ = hidden_units_;

    int isConfigExist = access("decoding_gemm_config.in", 0);
    if (isConfigExist == -1) {
      printf("[WARNING] decoding_gemm_config.in is not found\n");
    } else {
      readAlgoFromConfig(cublasAlgoMap_);
      // check that the gemm_config setting is runnable
      for (auto iter = cublasAlgoMap_.begin(); iter != cublasAlgoMap_.end();
           iter++) {
        int algoId = iter->second.algoId;
        int stages = iter->second.stages;
        // only check for cublas
        if (stages != -1) continue;
        if (Traits_::OpType == OperationType::FP32) {
          if (algoId > CUBLAS_GEMM_ALGO23 || algoId < CUBLAS_GEMM_DEFAULT) {
            // the algorithm is not for FP32
            printf("[ERROR] cuBLAS Algorithm %d is not used in FP32. \n",
                   algoId);
            exit(-1);
          }
        } else {
          if (algoId > CUBLAS_GEMM_ALGO15_TENSOR_OP ||
              algoId < CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
            // the algorithm is not for FP16
            printf("[ERROR] cuBLAS Algorithm %d is not used in FP16. \n",
                   algoId);
            exit(-1);
          }
        }
      }
    }
    judgeFusedQKV();
  }

  inline void set_max_batch_size(int batch_size) {
    max_batch_size_ = batch_size;
  }

  int getWorkspaceSize() {
    assert(max_batch_size_ != -1);
    return 13 * max_batch_size_ * hidden_units_ + sizeof(DataType_ *) * 9;
  }

  void set_tensor_parallel_param(const TensorParallelParam param) {
    t_parallel_param_ = param;
  }

  void set_layer_parallel_param(const LayerParallelParam param) {
    l_parallel_param_ = param;
  }

  void initialize(DecoderInitParam<DataType_> param,
                  DataType_ *buf,
                  void *cublas_workapsce,
                  bool set_local_batch = true) {
#ifndef NDEBUG
// PRINT_FUNC_NAME_();
#endif
    param_ = param;
    if (l_parallel_param_.local_batch_size == -1 || set_local_batch == true)
      l_parallel_param_.local_batch_size = param_.request_batch_size;
    const int buf_size = max_batch_size_ * hidden_units_;
    // cublas_workspace_ should be the start pointer of cudaMalloc()
    // to ensure 16B alignemnet
    cublas_workspace_ = cublas_workapsce;

    norm_from_tensor_buf_ = buf;
    ffn_out_buf_ = buf;
    query_buf_ = buf + buf_size;  // store the query values (from_tensor * Q) in
                                  // both masked and multi-head attention
    key_buf_ = query_buf_ + buf_size;
    value_buf_ = key_buf_ + buf_size;
    context_buf_ = value_buf_ + buf_size;  // store the context result
                                           // (softmax(qk)v) in both masked and
                                           // multi-head attention

    masked_output_buf_ = context_buf_ + buf_size;  // masked_attention_output
    norm_masked_output_buf_ =
        masked_output_buf_ + buf_size;  // norm(masked_attention_output)

    cross_output_buf_ =
        norm_masked_output_buf_ + buf_size;  // mutli-head attention_output
    norm_cross_output_buf_ =
        cross_output_buf_ + buf_size;  // norm(multi-head attention_output)
    ffn_inner_buf_ =
        norm_cross_output_buf_ + buf_size;  // 4 buf size to store inner product

    qkv_kernel_ = (DataType_ **)(ffn_inner_buf_ + 4 * buf_size);
    qkv_input_ = qkv_kernel_ + 3;
    qkv_buf_ = qkv_input_ + 3;

    if (is_fuse_QKV_in_normal_gemm_ == false &&
        is_fuse_QKV_in_batched_gemm_ == true) {
      const DataType_ *hA[]{param_.self_attention.query_weight.kernel,
                            param_.self_attention.key_weight.kernel,
                            param_.self_attention.value_weight.kernel,
                            norm_from_tensor_buf_,
                            norm_from_tensor_buf_,
                            norm_from_tensor_buf_,
                            query_buf_,
                            key_buf_,
                            value_buf_};
      cudaMemcpyAsync((void *)qkv_kernel_,
                      hA,
                      sizeof(DataType_ *) * 9,
                      cudaMemcpyHostToDevice,
                      param_.stream);
    }
  }

  void forward(const DataType_ *from_tensor,
               const DataType_ *memory_tensor,
               DataType_ *key_cache_,
               DataType_ *value_cache_,
               DataType_ *key_mem_cache_,
               DataType_ *value_mem_cache_,
               const int *memory_sequence_length,
               DataType_ *decoder_output,
               const int step,
               const int decoder_max_seq_len,
               const bool is_cross_attention,
               const bool *finished = nullptr,
               const int memory_max_seq_len = -1) {
#ifndef NDEBUG
// PRINT_FUNC_NAME_();
#endif
    const int m = l_parallel_param_.local_batch_size;

    try {
      /* masked multi-head attention */
      /* layernorm(from_tensor) -> norm_from_tensor_buf_ */
      if (normalization_before_) {
        layer_norm(from_tensor,
                   param_.self_layernorm.gamma,
                   param_.self_layernorm.beta,
                   norm_from_tensor_buf_,
                   m,
                   hidden_units_,
                   param_.stream);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        PUSH_RANGE("Transformer/slf_attn")
        if (memory_max_seq_len == -1) {
          masked_multi_head_attention(norm_from_tensor_buf_,
                                      key_cache_,
                                      value_cache_,
                                      masked_output_buf_,
                                      finished,
                                      step,
                                      decoder_max_seq_len);
        } else {
          self_multi_head_attention(norm_from_tensor_buf_,
                                    memory_sequence_length,
                                    key_cache_,
                                    value_cache_,
                                    masked_output_buf_,
                                    step + memory_max_seq_len,
                                    memory_max_seq_len);
        }
        POP_RANGE

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        if (is_cross_attention == true) {
          /*
              add bias to masked_output_buf_
              masked_output_buf_ + from_tensor -> masked_output_buf_
              norm(masked_output_buf_) -> norm_masked_output_buf_
          */
          add_bias_input_layernorm_2_kernelLauncher(
              from_tensor,
              param_.cross_layernorm.gamma,
              param_.cross_layernorm.beta,
              param_.self_attention.attention_output_weight.bias,
              masked_output_buf_,
              norm_masked_output_buf_,
              m,
              hidden_units_,
              param_.stream);
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          // For Attention is All You Need decoder
          /* cross attention with memory */
          cross_multi_head_attention(norm_masked_output_buf_,
                                     memory_tensor,
                                     key_mem_cache_,
                                     value_mem_cache_,
                                     cross_output_buf_,
                                     memory_sequence_length,
                                     finished,
                                     param_.request_max_mem_seq_len,
                                     step);
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          /*
              cross_output_buf_ + bias + masked_output_buf_ -> cross_output_buf_
              norm(cross_otuput_buf) -> normed_last_context (input for ffn)
          */
          add_bias_input_layernorm_2_kernelLauncher(
              masked_output_buf_,
              param_.ffn_layernorm.gamma,
              param_.ffn_layernorm.beta,
              param_.cross_attention.attention_output_weight.bias,
              cross_output_buf_,
              norm_cross_output_buf_,
              m,
              hidden_units_,
              param_.stream);
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          ffn(norm_cross_output_buf_,
              ffn_inner_buf_,
              decoder_output,
              m,
              4 * t_parallel_param_.local_hidden_units_,
              hidden_units_,
              act_);
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          add_bias_input_kernelLauncher(decoder_output,
                                        param_.ffn.output_weight.bias,
                                        cross_output_buf_,
                                        m,
                                        hidden_units_,
                                        param_.stream);
        } else {
          add_bias_input_layernorm_2_kernelLauncher(
              from_tensor,
              param_.ffn_layernorm.gamma,
              param_.ffn_layernorm.beta,
              param_.self_attention.attention_output_weight.bias,
              masked_output_buf_,
              norm_masked_output_buf_,
              m,
              hidden_units_,
              param_.stream);
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          // For GPT-2 decoder
          PUSH_RANGE("Transformer/MLP")
          ffn(norm_masked_output_buf_,
              ffn_inner_buf_,
              decoder_output,
              m,
              4 * t_parallel_param_.local_hidden_units_,
              hidden_units_,
              act_);
          POP_RANGE
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          add_bias_input_kernelLauncher(decoder_output,
                                        param_.ffn.output_weight.bias,
                                        masked_output_buf_,
                                        m,
                                        hidden_units_,
                                        param_.stream);
        }
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
      } else {
        // post-normalization
        if (memory_max_seq_len == -1) {
          masked_multi_head_attention(from_tensor,
                                      key_cache_,
                                      value_cache_,
                                      masked_output_buf_,
                                      finished,
                                      step,
                                      decoder_max_seq_len);
        } else {
          self_multi_head_attention(from_tensor,
                                    memory_sequence_length,
                                    key_cache_,
                                    value_cache_,
                                    masked_output_buf_,
                                    step + memory_max_seq_len,
                                    memory_max_seq_len);
        }

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        add_bias_input_layernorm_2_kernelLauncher(
            from_tensor,
            param_.self_layernorm.gamma,
            param_.self_layernorm.beta,
            param_.self_attention.attention_output_weight.bias,
            masked_output_buf_,
            norm_masked_output_buf_,
            m,
            hidden_units_,
            param_.stream);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        if (is_cross_attention == true) {
          // For Attention is All You Need decoder
          // cross attention with memory
          cross_multi_head_attention(norm_masked_output_buf_,
                                     memory_tensor,
                                     key_mem_cache_,
                                     value_mem_cache_,
                                     cross_output_buf_,
                                     memory_sequence_length,
                                     finished,
                                     param_.request_max_mem_seq_len,
                                     step);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          //
          //  cross_output_buf_ + bias + masked_output_buf_ -> cross_output_buf_
          //  norm(cross_otuput_buf) -> normed_last_context (input for ffn)
          //
          add_bias_input_layernorm_2_kernelLauncher(
              norm_masked_output_buf_,
              param_.cross_layernorm.gamma,
              param_.cross_layernorm.beta,
              param_.cross_attention.attention_output_weight.bias,
              cross_output_buf_,
              norm_cross_output_buf_,
              m,
              hidden_units_,
              param_.stream);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          ffn(norm_cross_output_buf_,
              ffn_inner_buf_,
              ffn_out_buf_,
              m,
              4 * t_parallel_param_.local_hidden_units_,
              hidden_units_,
              act_);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          add_bias_input_kernelLauncher(ffn_out_buf_,
                                        param_.ffn.output_weight.bias,
                                        norm_cross_output_buf_,
                                        m,
                                        hidden_units_,
                                        param_.stream);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif

          layer_norm(ffn_out_buf_,
                     param_.ffn_layernorm.gamma,
                     param_.ffn_layernorm.beta,
                     decoder_output,
                     m,
                     hidden_units_,
                     param_.stream);

        } else {
          ffn(norm_masked_output_buf_,
              ffn_inner_buf_,
              ffn_out_buf_,
              m,
              4 * t_parallel_param_.local_hidden_units_,
              hidden_units_,
              act_);
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif

          add_bias_input_kernelLauncher(ffn_out_buf_,
                                        param_.ffn.output_weight.bias,
                                        norm_masked_output_buf_,
                                        m,
                                        hidden_units_,
                                        param_.stream);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif

          layer_norm(ffn_out_buf_,
                     param_.ffn_layernorm.gamma,
                     param_.ffn_layernorm.beta,
                     decoder_output,
                     m,
                     hidden_units_,
                     param_.stream);
        }
      }
    } catch (std::runtime_error &error) {
      throw error;
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
  }


  void forward_v2(const DataType_ *from_tensor,
                  const DataType_ *memory_tensor,
                  DataType_ *key_cache_,
                  DataType_ *value_cache_,
                  DataType_ *key_mem_cache_,
                  DataType_ *value_mem_cache_,
                  const int *memory_sequence_length,
                  DataType_ *decoder_output,
                  const int step,
                  const int decoder_max_seq_len,
                  const bool is_cross_attention,
                  const bool *finished = nullptr,
                  const int max_input_len = 0,
                  const int *input_lengths = nullptr) {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    const int m = l_parallel_param_.local_batch_size;
    try {
      /* masked multi-head attention */
      /* layernorm(from_tensor) -> norm_from_tensor_buf_ */
      if (normalization_before_) {
        layer_norm(from_tensor,
                   param_.self_layernorm.gamma,
                   param_.self_layernorm.beta,
                   norm_from_tensor_buf_,
                   m,
                   hidden_units_,
                   param_.stream);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        PUSH_RANGE("Transformer/slf_attn")
        masked_multi_head_attention_v2(norm_from_tensor_buf_,
                                       key_cache_,
                                       value_cache_,
                                       masked_output_buf_,
                                       finished,
                                       step,
                                       decoder_max_seq_len,
                                       max_input_len,
                                       input_lengths);
        POP_RANGE

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        if (is_cross_attention == true) {
          /*
              add bias to masked_output_buf_
              masked_output_buf_ + from_tensor -> masked_output_buf_
              norm(masked_output_buf_) -> norm_masked_output_buf_
          */
          add_bias_input_layernorm_2_kernelLauncher(
              from_tensor,
              param_.cross_layernorm.gamma,
              param_.cross_layernorm.beta,
              param_.self_attention.attention_output_weight.bias,
              masked_output_buf_,
              norm_masked_output_buf_,
              m,
              hidden_units_,
              param_.stream);
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          // For Attention is All You Need decoder
          /* cross attention with memory */
          cross_multi_head_attention(norm_masked_output_buf_,
                                     memory_tensor,
                                     key_mem_cache_,
                                     value_mem_cache_,
                                     cross_output_buf_,
                                     memory_sequence_length,
                                     finished,
                                     param_.request_max_mem_seq_len,
                                     step);
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          /*
              cross_output_buf_ + bias + masked_output_buf_ -> cross_output_buf_
              norm(cross_otuput_buf) -> normed_last_context (input for ffn)
          */
          add_bias_input_layernorm_2_kernelLauncher(
              masked_output_buf_,
              param_.ffn_layernorm.gamma,
              param_.ffn_layernorm.beta,
              param_.cross_attention.attention_output_weight.bias,
              cross_output_buf_,
              norm_cross_output_buf_,
              m,
              hidden_units_,
              param_.stream);
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          ffn(norm_cross_output_buf_,
              ffn_inner_buf_,
              decoder_output,
              m,
              4 * t_parallel_param_.local_hidden_units_,
              hidden_units_,
              ActivationType::RELU);
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          add_bias_input_kernelLauncher(decoder_output,
                                        param_.ffn.output_weight.bias,
                                        cross_output_buf_,
                                        m,
                                        hidden_units_,
                                        param_.stream);
        } else {
          add_bias_input_layernorm_2_kernelLauncher(
              from_tensor,
              param_.ffn_layernorm.gamma,
              param_.ffn_layernorm.beta,
              param_.self_attention.attention_output_weight.bias,
              masked_output_buf_,
              norm_masked_output_buf_,
              m,
              hidden_units_,
              param_.stream);
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          PUSH_RANGE("Transformer/MLP")
          ffn(norm_masked_output_buf_,
              ffn_inner_buf_,
              decoder_output,
              m,
              4 * t_parallel_param_.local_hidden_units_,
              hidden_units_,
              ActivationType::GELU);
          POP_RANGE

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          add_bias_input_kernelLauncher(decoder_output,
                                        param_.ffn.output_weight.bias,
                                        masked_output_buf_,
                                        m,
                                        hidden_units_,
                                        param_.stream);
        }
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
      } else {
        // post-normalization
        PUSH_RANGE("Transformer/slf_attn")
        masked_multi_head_attention_v2(from_tensor,
                                       key_cache_,
                                       value_cache_,
                                       masked_output_buf_,
                                       finished,
                                       step,
                                       decoder_max_seq_len,
                                       max_input_len,
                                       input_lengths);
        POP_RANGE

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        add_bias_input_layernorm_2_kernelLauncher(
            from_tensor,
            param_.self_layernorm.gamma,
            param_.self_layernorm.beta,
            param_.self_attention.attention_output_weight.bias,
            masked_output_buf_,
            norm_masked_output_buf_,
            m,
            hidden_units_,
            param_.stream);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        if (is_cross_attention == true) {
          // For Attention is All You Need decoder
          /* cross attention with memory */
          cross_multi_head_attention(norm_masked_output_buf_,
                                     memory_tensor,
                                     key_mem_cache_,
                                     value_mem_cache_,
                                     cross_output_buf_,
                                     memory_sequence_length,
                                     finished,
                                     param_.request_max_mem_seq_len,
                                     step);


#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          /*
              cross_output_buf_ + bias + masked_output_buf_ -> cross_output_buf_
              norm(cross_otuput_buf) -> normed_last_context (input for ffn)
          */
          add_bias_input_layernorm_2_kernelLauncher(
              norm_masked_output_buf_,
              param_.cross_layernorm.gamma,
              param_.cross_layernorm.beta,
              param_.cross_attention.attention_output_weight.bias,
              cross_output_buf_,
              norm_cross_output_buf_,
              m,
              hidden_units_,
              param_.stream);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
          ffn(norm_cross_output_buf_,
              ffn_inner_buf_,
              ffn_out_buf_,
              m,
              4 * t_parallel_param_.local_hidden_units_,
              hidden_units_,
              act_);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif

          add_bias_input_kernelLauncher(ffn_out_buf_,
                                        param_.ffn.output_weight.bias,
                                        norm_cross_output_buf_,
                                        m,
                                        hidden_units_,
                                        param_.stream);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif

          layer_norm(ffn_out_buf_,
                     param_.ffn_layernorm.gamma,
                     param_.ffn_layernorm.beta,
                     decoder_output,
                     m,
                     hidden_units_,
                     param_.stream);
        } else {
          PUSH_RANGE("Transformer/MLP")
          ffn(norm_masked_output_buf_,
              ffn_inner_buf_,
              ffn_out_buf_,
              m,
              4 * t_parallel_param_.local_hidden_units_,
              hidden_units_,
              act_);
          POP_RANGE

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif

          add_bias_input_kernelLauncher(ffn_out_buf_,
                                        param_.ffn.output_weight.bias,
                                        norm_masked_output_buf_,
                                        m,
                                        hidden_units_,
                                        param_.stream);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif

          layer_norm(ffn_out_buf_,
                     param_.ffn_layernorm.gamma,
                     param_.ffn_layernorm.beta,
                     decoder_output,
                     m,
                     hidden_units_,
                     param_.stream);
        }
      }
    } catch (std::runtime_error &error) {
      throw error;
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
  }

  size_t getContextWorkspaceSize(const int seq_len,
                                 const int local_batch_size) {
    const size_t m = local_batch_size * seq_len;
    const size_t qk_buf_size =
        (size_t)(ceil(local_batch_size * t_parallel_param_.local_head_num_ *
                      seq_len * seq_len / 4.)) *
        4;
    const size_t attn_work_space_size =
        3 * m * hidden_units_ /* Q, K, V */ +
        3 * m *
            t_parallel_param_.local_hidden_units_ /* q_buf, k_buf, v_buf */ +
        qk_buf_size +
        2 * m * t_parallel_param_.local_hidden_units_ /* trans_attn, attn */;
    return (m * hidden_units_ * 3 + attn_work_space_size +
            m * t_parallel_param_.local_hidden_units_ * 4 /* ffn buffer */) *
           sizeof(DataType_);
  }

  // use to compute the context of gpt model
  void forward_context(DataType_ *workspace,
                       DataType_ *decoder_output,
                       DataType_ *key_cache_,
                       DataType_ *value_cache_,
                       const DataType_ *from_tensor,
                       const DataType_ *d_attn_mask,
                       const int local_batch_size,
                       const int seq_len,
                       const int ite,
                       const int max_seq_len,
                       const bool is_final) {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    try {
      const int m = local_batch_size * seq_len;
      const int qk_buf_size =
          (int)(ceil(local_batch_size * t_parallel_param_.local_head_num_ *
                     seq_len * seq_len / 4.)) *
          4;
      const int attn_work_space_size =
          3 * m * hidden_units_ /* Q, K, V */ +
          3 * m *
              t_parallel_param_.local_hidden_units_ /* q_buf, k_buf, v_buf */ +
          qk_buf_size +
          2 * m * t_parallel_param_.local_hidden_units_ /* trans_attn, attn */;

      // set workspace
      DataType_ *norm_from_tensor_buf = (DataType_ *)workspace;
      DataType_ *attention_workspace = norm_from_tensor_buf + m * hidden_units_;
      DataType_ *masked_output_buf = attention_workspace + attn_work_space_size;
      DataType_ *norm_masked_output_buf = masked_output_buf + m * hidden_units_;
      DataType_ *ffn_inner_buf = norm_masked_output_buf + m * hidden_units_;

      layer_norm(from_tensor,
                 param_.self_layernorm.gamma,
                 param_.self_layernorm.beta,
                 norm_from_tensor_buf,
                 m,
                 hidden_units_,
                 param_.stream);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
      PUSH_RANGE("Transformer/slf_attn")
      unfused_masked_multi_head_attention(attention_workspace,
                                          norm_from_tensor_buf,
                                          key_cache_,
                                          value_cache_,
                                          masked_output_buf,
                                          d_attn_mask,
                                          local_batch_size,
                                          seq_len,
                                          ite,
                                          max_seq_len,
                                          is_final);
      if (is_final) return;
      POP_RANGE
#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
      add_bias_input_layernorm_2_kernelLauncher(
          from_tensor,
          param_.ffn_layernorm.gamma,
          param_.ffn_layernorm.beta,
          param_.self_attention.attention_output_weight.bias,
          masked_output_buf,
          norm_masked_output_buf,
          m,
          hidden_units_,
          param_.stream);
#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
      // For GPT decoder
      PUSH_RANGE("Transformer/MLP");
      ffn(norm_masked_output_buf,
          ffn_inner_buf,
          decoder_output,
          m,
          4 * t_parallel_param_.local_hidden_units_,
          hidden_units_,
          ActivationType::GELU);
      POP_RANGE
#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
      add_bias_input_kernelLauncher(decoder_output,
                                    param_.ffn.output_weight.bias,
                                    masked_output_buf,
                                    m,
                                    hidden_units_,
                                    param_.stream);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
    } catch (std::runtime_error &error) {
      throw error;
    }
  }

  void self_multi_head_attention(const DataType_ *from_tensor,
                                 const int *memory_sequence_length,
                                 DataType_ *key_cache_,
                                 DataType_ *value_cache_,
                                 DataType_ *decoder_output,
                                 const int step,
                                 const int memory_max_seq_len) {
    int m = l_parallel_param_.local_batch_size;
    int n = t_parallel_param_.local_hidden_units_;
    int k = hidden_units_;


    DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;
    if (is_fuse_QKV_in_batched_gemm_ == true) {
      cublasGemmAlgo_t cublasAlgo =
          static_cast<cublasGemmAlgo_t>(getAlgoIdFromMap(
              cublasAlgoMap_,
              3,
              n,
              m,
              k,
              std::is_same<float, DataType_>::value ? FLOAT_DATATYPE
                                                    : HALF_DATATYPE));
      check_cuda_error(cublasGemmBatchedEx(param_.cublas_handle,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_N,
                                           n,
                                           m,
                                           k,
                                           &alpha,
                                           (const void *const *)qkv_kernel_,
                                           AType_,
                                           n,
                                           (const void *const *)qkv_input_,
                                           BType_,
                                           k,
                                           &beta,
                                           (void *const *)qkv_buf_,
                                           CType_,
                                           n,
                                           3,
                                           computeType_,
                                           cublasAlgo));
    } else {
      cublasMM_cublasLtMM_wrapper_decoder(
          param_.cublaslt_handle,
          param_.cublas_handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          m,
          k,
          &alpha,
          param_.self_attention.query_weight.kernel,
          AType_,
          n,
          from_tensor,
          BType_,
          k,
          &beta,
          query_buf_,
          CType_,
          n,
          param_.stream,
          cublasAlgoMap_,
          cublas_workspace_);

      cublasMM_cublasLtMM_wrapper_decoder(
          param_.cublaslt_handle,
          param_.cublas_handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          m,
          k,
          &alpha,
          param_.self_attention.key_weight.kernel,
          AType_,
          n,
          from_tensor,
          BType_,
          k,
          &beta,
          key_buf_,
          CType_,
          n,
          param_.stream,
          cublasAlgoMap_,
          cublas_workspace_);

      cublasMM_cublasLtMM_wrapper_decoder(
          param_.cublaslt_handle,
          param_.cublas_handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          m,
          k,
          &alpha,
          param_.self_attention.value_weight.kernel,
          AType_,
          n,
          from_tensor,
          BType_,
          k,
          &beta,
          value_buf_,
          CType_,
          n,
          param_.stream,
          cublasAlgoMap_,
          cublas_workspace_);
    }
    self_attention_dispatch<DataType_>(memory_sequence_length,
                                       key_buf_,
                                       value_buf_,
                                       query_buf_,
                                       param_.self_attention.query_weight.bias,
                                       key_cache_,
                                       param_.self_attention.key_weight.bias,
                                       value_cache_,
                                       param_.self_attention.value_weight.bias,
                                       context_buf_,
                                       param_.request_batch_size,
                                       head_num_,
                                       size_per_head_,
                                       step,
                                       memory_max_seq_len,
                                       param_.stream);


    k = t_parallel_param_.local_hidden_units_;
    n = hidden_units_;

    cublasMM_cublasLtMM_wrapper_decoder(
        param_.cublaslt_handle,
        param_.cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        param_.self_attention.attention_output_weight.kernel,
        AType_,
        n,
        context_buf_,
        BType_,
        k,
        &beta,
        decoder_output,
        CType_,
        n,
        param_.stream,
        cublasAlgoMap_,
        cublas_workspace_);

    PUSH_RANGE("Transformer/slf_attn/all2all_reduce")
    all2all_reduce_sum(decoder_output,
                       decoder_output,
                       m * n,
                       t_parallel_param_,
                       param_.stream);
    POP_RANGE
  }
  void masked_multi_head_attention(const DataType_ *from_tensor,
                                   DataType_ *key_cache_,
                                   DataType_ *value_cache_,
                                   DataType_ *decoder_output,
                                   const bool *finished,
                                   const int step,
                                   const int max_seq_len) {
    int m = l_parallel_param_.local_batch_size;
    int n = t_parallel_param_.local_hidden_units_;
    int k = hidden_units_;

    // chose which attention to use
    int decoder_max_seq_len = (getCacheFormat() != 0) ? max_seq_len : -1;

    DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

    if (is_fuse_QKV_in_normal_gemm_ == true) {
      cublasMM_cublasLtMM_wrapper_decoder(
          param_.cublaslt_handle,
          param_.cublas_handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          3 * n,
          m,
          k,
          &alpha,
          param_.self_attention.query_weight.kernel,
          AType_,
          3 * n,
          from_tensor,
          BType_,
          k,
          &beta,
          query_buf_,
          CType_,
          3 * n,
          param_.stream,
          cublasAlgoMap_,
          cublas_workspace_);

      fusedQKV_masked_attention_dispatch<DataType_>(
          query_buf_,
          param_.self_attention.query_weight.bias,
          key_cache_,
          value_cache_,
          context_buf_,
          finished,
          param_.request_batch_size,
          l_parallel_param_.local_batch_size,
          t_parallel_param_.local_head_num_,
          size_per_head_,
          step,
          decoder_max_seq_len,
          param_.stream);
    } else {
      if (is_fuse_QKV_in_batched_gemm_ == true) {
        cublasGemmAlgo_t cublasAlgo =
            static_cast<cublasGemmAlgo_t>(getAlgoIdFromMap(
                cublasAlgoMap_,
                3,
                n,
                m,
                k,
                std::is_same<float, DataType_>::value ? FLOAT_DATATYPE
                                                      : HALF_DATATYPE));
        check_cuda_error(cublasGemmBatchedEx(param_.cublas_handle,
                                             CUBLAS_OP_N,
                                             CUBLAS_OP_N,
                                             n,
                                             m,
                                             k,
                                             &alpha,
                                             (const void *const *)qkv_kernel_,
                                             AType_,
                                             n,
                                             (const void *const *)qkv_input_,
                                             BType_,
                                             k,
                                             &beta,
                                             (void *const *)qkv_buf_,
                                             CType_,
                                             n,
                                             3,
                                             computeType_,
                                             cublasAlgo));
      } else {
        cublasMM_cublasLtMM_wrapper_decoder(
            param_.cublaslt_handle,
            param_.cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            param_.self_attention.query_weight.kernel,
            AType_,
            n,
            from_tensor,
            BType_,
            k,
            &beta,
            query_buf_,
            CType_,
            n,
            param_.stream,
            cublasAlgoMap_,
            cublas_workspace_);

        cublasMM_cublasLtMM_wrapper_decoder(
            param_.cublaslt_handle,
            param_.cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            param_.self_attention.key_weight.kernel,
            AType_,
            n,
            from_tensor,
            BType_,
            k,
            &beta,
            key_buf_,
            CType_,
            n,
            param_.stream,
            cublasAlgoMap_,
            cublas_workspace_);

        cublasMM_cublasLtMM_wrapper_decoder(
            param_.cublaslt_handle,
            param_.cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            param_.self_attention.value_weight.kernel,
            AType_,
            n,
            from_tensor,
            BType_,
            k,
            &beta,
            value_buf_,
            CType_,
            n,
            param_.stream,
            cublasAlgoMap_,
            cublas_workspace_);
      }
      masked_attention_dispatch<DataType_>(
          key_buf_,
          value_buf_,
          query_buf_,
          param_.self_attention.query_weight.bias,
          key_cache_,
          param_.self_attention.key_weight.bias,
          value_cache_,
          param_.self_attention.value_weight.bias,
          context_buf_,
          finished,
          param_.request_batch_size,
          l_parallel_param_.local_batch_size,
          t_parallel_param_.local_head_num_,
          size_per_head_,
          step,
          decoder_max_seq_len,
          param_.stream);
    }

    k = t_parallel_param_.local_hidden_units_;
    n = hidden_units_;

    cublasMM_cublasLtMM_wrapper_decoder(
        param_.cublaslt_handle,
        param_.cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        param_.self_attention.attention_output_weight.kernel,
        AType_,
        n,
        context_buf_,
        BType_,
        k,
        &beta,
        decoder_output,
        CType_,
        n,
        param_.stream,
        cublasAlgoMap_,
        cublas_workspace_);

    PUSH_RANGE("Transformer/slf_attn/all2all_reduce")
    all2all_reduce_sum(decoder_output,
                       decoder_output,
                       m * n,
                       t_parallel_param_,
                       param_.stream);
    POP_RANGE
  }

  void masked_multi_head_attention_v2(const DataType_ *from_tensor,
                                      DataType_ *key_cache_,
                                      DataType_ *value_cache_,
                                      DataType_ *decoder_output,
                                      const bool *finished,
                                      const int step,
                                      const int max_seq_len,
                                      const int max_input_len,
                                      const int *input_lengths) {
    assert(is_fuse_QKV_in_normal_gemm_ ==
           true);  // only support for is_fuse_QKV = True.

    int m = l_parallel_param_.local_batch_size;
    int n = t_parallel_param_.local_hidden_units_;
    int k = hidden_units_;

    assert(getCacheFormat() !=
           0);  // this is the only difference with masked_multi_head_attention

    DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

    cublasMM_cublasLtMM_wrapper_decoder(
        param_.cublaslt_handle,
        param_.cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        3 * n,
        m,
        k,
        &alpha,
        param_.self_attention.query_weight.kernel,
        AType_,
        3 * n,
        from_tensor,
        BType_,
        k,
        &beta,
        query_buf_,
        CType_,
        3 * n,
        param_.stream,
        cublasAlgoMap_,
        cublas_workspace_);

    fusedQKV_masked_attention_dispatch_v2<DataType_>(
        query_buf_,
        param_.self_attention.query_weight.bias,
        key_cache_,
        value_cache_,
        context_buf_,
        finished,
        param_.request_batch_size,
        l_parallel_param_.local_batch_size,
        t_parallel_param_.local_head_num_,
        size_per_head_,
        step,
        max_seq_len,
        max_input_len,
        input_lengths,
        param_.stream);

    k = t_parallel_param_.local_hidden_units_;
    n = hidden_units_;

    cublasMM_cublasLtMM_wrapper_decoder(
        param_.cublaslt_handle,
        param_.cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        param_.self_attention.attention_output_weight.kernel,
        AType_,
        n,
        context_buf_,
        BType_,
        k,
        &beta,
        decoder_output,
        CType_,
        n,
        param_.stream,
        cublasAlgoMap_,
        cublas_workspace_);

    PUSH_RANGE("Transformer/slf_attn/all2all_reduce")
    all2all_reduce_sum(decoder_output,
                       decoder_output,
                       m * n,
                       t_parallel_param_,
                       param_.stream);
    POP_RANGE
  }

  /* attention with source sentence */
  void cross_multi_head_attention(const DataType_ *from_tensor,
                                  const DataType_ *memory_tensor,
                                  DataType_ *key_mem_cache_,
                                  DataType_ *value_mem_cache_,
                                  DataType_ *decoder_output,
                                  const int *memory_sequence_length,
                                  const bool *finished,
                                  const int max_seq_len,
                                  const int step) {
    int m = param_.request_batch_size;
    int n = t_parallel_param_.local_hidden_units_;
    int k = hidden_units_;

    DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

    // reuse the query_buf
    cublasMM_cublasLtMM_wrapper_decoder(
        param_.cublaslt_handle,
        param_.cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        param_.cross_attention.query_weight.kernel,
        AType_,
        n,
        from_tensor,
        BType_,
        k,
        &beta,
        query_buf_,
        CType_,
        n,
        param_.stream,
        cublasAlgoMap_,
        cublas_workspace_);

    if (step == 1) {
      m *= max_seq_len;
      k = memory_hidden_units_;

      cublasMM_cublasLtMM_wrapper_decoder(
          param_.cublaslt_handle,
          param_.cublas_handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          m,
          k,
          &alpha,
          param_.cross_attention.key_weight.kernel,
          AType_,
          n,
          memory_tensor,
          BType_,
          k,
          &beta,
          key_mem_cache_,
          CType_,
          n,
          param_.stream,
          cublasAlgoMap_,
          cublas_workspace_);

      cublasMM_cublasLtMM_wrapper_decoder(
          param_.cublaslt_handle,
          param_.cublas_handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          m,
          k,
          &alpha,
          param_.cross_attention.value_weight.kernel,
          AType_,
          n,
          memory_tensor,
          BType_,
          k,
          &beta,
          value_mem_cache_,
          CType_,
          n,
          param_.stream,
          cublasAlgoMap_,
          cublas_workspace_);

      k = t_parallel_param_.local_hidden_units_;
    }

    cross_attention_dispatch<DataType_>(
        query_buf_,
        param_.cross_attention.query_weight.bias,
        key_mem_cache_,
        param_.cross_attention.key_weight.bias,
        value_mem_cache_,
        param_.cross_attention.value_weight.bias,
        memory_sequence_length,
        context_buf_,
        finished,
        param_.request_batch_size,
        head_num_,
        size_per_head_,
        step,
        max_seq_len,
        param_.stream);

    m = param_.request_batch_size;
    n = hidden_units_;
    k = t_parallel_param_.local_hidden_units_;

    cublasMM_cublasLtMM_wrapper_decoder(
        param_.cublaslt_handle,
        param_.cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        param_.cross_attention.attention_output_weight.kernel,
        AType_,
        n,
        context_buf_,
        BType_,
        k,
        &beta,
        decoder_output,
        CType_,
        n,
        param_.stream,
        cublasAlgoMap_,
        cublas_workspace_);
  }


  void ffn(const DataType_ *input,
           DataType_ *ffn_inner,
           DataType_ *output,
           const int m,
           const int inner_size,
           const int n,
           ActivationType activation_type) {
    int m1 = m, k1 = n, n1 = inner_size;
    DataType_ alpha = (DataType_)1.0f;
    DataType_ beta = (DataType_)0.0f;

    cublasMM_cublasLtMM_wrapper_decoder(param_.cublaslt_handle,
                                        param_.cublas_handle,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        n1,
                                        m1,
                                        k1,
                                        &alpha,
                                        param_.ffn.intermediate_weight.kernel,
                                        AType_,
                                        n1,
                                        input,
                                        BType_,
                                        k1,
                                        &beta,
                                        ffn_inner,
                                        CType_,
                                        n1,
                                        param_.stream,
                                        cublasAlgoMap_,
                                        cublas_workspace_);

    add_bias_act_kernelLauncher(ffn_inner,
                                param_.ffn.intermediate_weight.bias,
                                m1,
                                inner_size,
                                activation_type,
                                param_.stream);

    int m2 = m, n2 = n, k2 = inner_size;
    cublasMM_cublasLtMM_wrapper_decoder(param_.cublaslt_handle,
                                        param_.cublas_handle,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        n2,
                                        m2,
                                        k2,
                                        &alpha,
                                        param_.ffn.output_weight.kernel,
                                        AType_,
                                        n2,
                                        ffn_inner,
                                        BType_,
                                        k2,
                                        &beta,
                                        output,
                                        CType_,
                                        n2,
                                        param_.stream,
                                        cublasAlgoMap_,
                                        cublas_workspace_);

    PUSH_RANGE("Transformer/MLP/all2all_reduce")
    all2all_reduce_sum(output, output, m * n, t_parallel_param_, param_.stream);
    POP_RANGE
  }

  void unfused_masked_multi_head_attention(DataType_ *workspace,
                                           const DataType_ *from_tensor,
                                           DataType_ *key_cache_,
                                           DataType_ *value_cache_,
                                           DataType_ *decoder_output,
                                           const DataType_ *attr_mask,
                                           const int local_batch_size,
                                           const int seq_len,
                                           const int ite,
                                           const int max_seq_len,
                                           const bool is_final) {
    const DataType_ scalar = 1 / sqrtf(size_per_head_ * 1.0f);
    const int m = local_batch_size * seq_len;

    const int qk_buf_size =
        (int)(ceil(local_batch_size * t_parallel_param_.local_head_num_ *
                   seq_len * seq_len / 4.)) *
        4;

    DataType_ *Q = workspace;
    DataType_ *K = Q + m * hidden_units_;
    DataType_ *V = K + m * hidden_units_;
    DataType_ *q_buf = V + m * hidden_units_;
    DataType_ *k_buf = q_buf + m * t_parallel_param_.local_hidden_units_;
    DataType_ *v_buf = k_buf + m * t_parallel_param_.local_hidden_units_;
    DataType_ *qk_buf = v_buf + m * t_parallel_param_.local_hidden_units_;
    DataType_ *attn_trans_out = qk_buf + qk_buf_size;
    DataType_ *attn_out =
        attn_trans_out + m * t_parallel_param_.local_hidden_units_;

    DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

    if (is_fuse_QKV_in_normal_gemm_ == true) {
      const int n = t_parallel_param_.local_hidden_units_;
      const int k = hidden_units_;
      cublasMM_cublasLtMM_wrapper_decoder(
          param_.cublaslt_handle,
          param_.cublas_handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          3 * n,
          m,
          k,
          &alpha,
          param_.self_attention.query_weight.kernel,
          AType_,
          3 * n,
          from_tensor,
          BType_,
          k,
          &beta,
          Q,
          CType_,
          3 * n,
          param_.stream,
          cublasAlgoMap_,
          cublas_workspace_);

      add_fusedQKV_bias_transpose_kernelLauncher(
          q_buf,
          k_buf,
          v_buf,
          Q,
          param_.self_attention.query_weight.bias,
          local_batch_size,
          seq_len,
          t_parallel_param_.local_head_num_,
          size_per_head_,
          param_.stream);
    } else {
      const int n = t_parallel_param_.local_hidden_units_;
      const int k = hidden_units_;
      cublasMM_cublasLtMM_wrapper_decoder(
          param_.cublaslt_handle,
          param_.cublas_handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          m,
          k,
          &alpha,
          param_.self_attention.query_weight.kernel,
          AType_,
          n,
          from_tensor,
          BType_,
          k,
          &beta,
          Q,
          CType_,
          n,
          param_.stream,
          cublasAlgoMap_,
          cublas_workspace_);

      cublasMM_cublasLtMM_wrapper_decoder(
          param_.cublaslt_handle,
          param_.cublas_handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          m,
          k,
          &alpha,
          param_.self_attention.key_weight.kernel,
          AType_,
          n,
          from_tensor,
          BType_,
          k,
          &beta,
          K,
          CType_,
          n,
          param_.stream,
          cublasAlgoMap_,
          cublas_workspace_);

      cublasMM_cublasLtMM_wrapper_decoder(
          param_.cublaslt_handle,
          param_.cublas_handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          m,
          k,
          &alpha,
          param_.self_attention.value_weight.kernel,
          AType_,
          n,
          from_tensor,
          BType_,
          k,
          &beta,
          V,
          CType_,
          n,
          param_.stream,
          cublasAlgoMap_,
          cublas_workspace_);

      add_QKV_bias_transpose_kernelLauncher(
          q_buf,
          k_buf,
          v_buf,
          Q,
          param_.self_attention.query_weight.bias,
          K,
          param_.self_attention.key_weight.bias,
          V,
          param_.self_attention.value_weight.bias,
          local_batch_size,
          seq_len,
          t_parallel_param_.local_head_num_,
          size_per_head_,
          param_.stream);
    }

    // !!! need to implement cget_cache_config
    if (max_seq_len == -1 || USE_CACHE_BATCH_MAJOR_ATTENTION == 0) {
      transpose_4d_kernelLauncher(key_cache_,
                                  k_buf,
                                  local_batch_size,
                                  seq_len,
                                  size_per_head_,
                                  t_parallel_param_.local_hidden_units_,
                                  t_parallel_param_.local_head_num_,
                                  param_.request_batch_size,
                                  ite,
                                  param_.stream);

      transpose_4d_kernelLauncher(value_cache_,
                                  v_buf,
                                  local_batch_size,
                                  seq_len,
                                  size_per_head_,
                                  t_parallel_param_.local_hidden_units_,
                                  t_parallel_param_.local_head_num_,
                                  param_.request_batch_size,
                                  ite,
                                  param_.stream);
    } else if (USE_CACHE_BATCH_MAJOR_ATTENTION == 1) {
      // Use batch major
      // put k/v_buf from shape [B, H, L, Dh]
      // to cache [B, H, Dh/x, L, x]  and [B, H, L, Dh/x, x]
      transpose_4d_batch_major_kernelLauncher(key_cache_,
                                              value_cache_,
                                              k_buf,
                                              v_buf,
                                              local_batch_size,
                                              seq_len,
                                              max_seq_len,
                                              size_per_head_,
                                              t_parallel_param_.local_head_num_,
                                              param_.stream);
    } else {
      printf("[ERROR] Can not decide on the cache config \n");
      exit(-1);
    }

    if (is_final) return;

    cublasGemmAlgo_t cublasAlgo =
        static_cast<cublasGemmAlgo_t>(getAlgoIdFromMap(
            cublasAlgoMap_,
            local_batch_size * t_parallel_param_.local_head_num_,
            seq_len,
            seq_len,
            size_per_head_,
            std::is_same<float, DataType_>::value ? FLOAT_DATATYPE
                                                  : HALF_DATATYPE));

    check_cuda_error(cublasGemmStridedBatchedEx(
        param_.cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        seq_len,
        seq_len,
        size_per_head_,
        &alpha,
        k_buf,
        AType_,
        size_per_head_,
        seq_len * size_per_head_,
        q_buf,
        BType_,
        size_per_head_,
        seq_len * size_per_head_,
        &beta,
        qk_buf,
        CType_,
        seq_len,
        seq_len * seq_len,
        local_batch_size * t_parallel_param_.local_head_num_,
        computeType_,
        cublasAlgo));

    attn_softmax_kernelLauncher(qk_buf,
                                attr_mask,
                                local_batch_size,
                                seq_len,
                                t_parallel_param_.local_head_num_,
                                scalar,
                                param_.stream);

    cublasAlgo = static_cast<cublasGemmAlgo_t>(getAlgoIdFromMap(
        cublasAlgoMap_,
        local_batch_size * t_parallel_param_.local_head_num_,
        size_per_head_,
        seq_len,
        seq_len,
        std::is_same<float, DataType_>::value ? FLOAT_DATATYPE
                                              : HALF_DATATYPE));

    check_cuda_error(cublasGemmStridedBatchedEx(
        param_.cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        size_per_head_,
        seq_len,
        seq_len,
        &alpha,
        v_buf,
        AType_,
        size_per_head_,
        seq_len * size_per_head_,
        qk_buf,
        BType_,
        seq_len,
        seq_len * seq_len,
        &beta,
        attn_trans_out,
        CType_,
        size_per_head_,
        seq_len * size_per_head_,
        local_batch_size * t_parallel_param_.local_head_num_,
        computeType_,
        cublasAlgo));

    transpose_kernelLauncher(attn_out,
                             attn_trans_out,
                             local_batch_size,
                             seq_len,
                             t_parallel_param_.local_head_num_,
                             size_per_head_,
                             param_.stream);

    {
      const int k = t_parallel_param_.local_hidden_units_;
      const int n = hidden_units_;

      cublasMM_cublasLtMM_wrapper_decoder(
          param_.cublaslt_handle,
          param_.cublas_handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          m,
          k,
          &alpha,
          param_.self_attention.attention_output_weight.kernel,
          AType_,
          n,
          attn_out,
          BType_,
          k,
          &beta,
          decoder_output,
          CType_,
          n,
          param_.stream,
          cublasAlgoMap_,
          cublas_workspace_);

      PUSH_RANGE("Transformer/slf_attn/all2all_reduce")
      all2all_reduce_sum(decoder_output,
                         decoder_output,
                         m * n,
                         t_parallel_param_,
                         param_.stream);
      POP_RANGE
    }
  }

  int getCacheFormat() {
    int x = (Traits_::OpType == OperationType::FP32) ? 4 : 8;
    return (USE_CACHE_BATCH_MAJOR_ATTENTION == 1 && size_per_head_ % x == 0)
               ? x
               : 0;
  }

  ~OpenDecoder() {
    norm_from_tensor_buf_ = nullptr;
    query_buf_ = nullptr;
    key_buf_ = nullptr;
    value_buf_ = nullptr;
    context_buf_ = nullptr;

    masked_output_buf_ = nullptr;
    norm_masked_output_buf_ = nullptr;

    cross_output_buf_ = nullptr;
    norm_cross_output_buf_ = nullptr;
    ffn_inner_buf_ = nullptr;
  }

  inline void set_local_batch_size(int local_batch) {
    l_parallel_param_.local_batch_size = local_batch;
  }
};
}  // namespace fastertransformer
