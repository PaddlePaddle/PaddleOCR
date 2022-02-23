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
 * Decoder transformer
 **/

#pragma once

#include <cuda_runtime.h>
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/open_decoder.h"
#include "fastertransformer/utils/allocator.h"
#include "fastertransformer/utils/arguments.h"
#include "fastertransformer/utils/common.h"
#include "fastertransformer/utils/functions.h"

namespace fastertransformer {

template <OperationType OpType_>
class DecodingBeamsearch {
private:
  typedef DecoderTransformerTraits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  const IAllocator &allocator_;
  struct DecodingBeamsearchArguments args_;

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  std::map<std::string, cublasLtMatmulAlgo_info> cublasAlgoMap_;

  OpenDecoder<OpType_> *decoder_;
  DataType_ **K_cache_;
  DataType_ **V_cache_;
  DataType_ **K_mem_cache_;
  DataType_ **V_mem_cache_;
  DataType_ *from_tensor_[2];
  DataType_ *decoder_buf_;

  // Prefix LM
  DataType_ *trans_out_buf_;
  DataType_ *lm_normed_result_buf_;

  DataType_ *decoder_normed_result_buf_;
  DataType_ *embedding_buf_;
  float *logits_buf_;
  float *cum_log_buf_;
  int *word_ids_buf_;
  int *parent_ids_buf_;
  bool *finished_buf_;
  bool *alive_finished_buf_;

  void *buf_;
  int *finished_count_buf_;
  bool *h_finished_buf_;
  int *h_trg_length_;
  float *temp_storage_;

  bool is_fuse_topk_softMax_;
  bool keep_alive_beam_;

  void *topK_kernel_workspace = nullptr;
  size_t topk_workspace_size_ = 0;
  void *cublas_workspace_ = nullptr;

  DataType_ *padded_embedding_kernel;
  DataType_ *padded_embedding_bias;
  DataType_ *tmp_logits_buf_;

public:
  DecodingBeamsearch(const IAllocator &allocator,
                     const int batch_size,
                     const int beam_width,
                     const int seq_len,
                     const int head_num,
                     const int size_per_head,
                     const int vocab_size,
                     const int decoder_layers,
                     const int memory_hidden_units,
                     const int memory_max_seq_len,
                     const int start_id,
                     const int end_id,
                     const float beam_search_diversity_rate = -0.0f,
                     const bool is_fuse_topk_softMax = false,
                     const bool is_fuse_qkv = false,
                     const bool keep_alive_beam = false,
                     const float alpha = 0.6,
                     const bool normalization_before = true,
                     const int pos_offset = 0,
                     const ActivationType act = ActivationType::RELU,
                     const bool pos_bias = false,
                     const bool prefix_lm = false,
                     const int finished_candidate_num = -1,
                     const bool early_stopping = false,
                     const bool is_mbart = false)
      : allocator_(allocator),
        is_fuse_topk_softMax_(is_fuse_topk_softMax),
        keep_alive_beam_(keep_alive_beam) {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    args_.batch_size_ = batch_size;
    args_.beam_width_ = beam_width;
    args_.seq_len_ = seq_len;
    args_.memory_max_seq_len_ = memory_max_seq_len;
    args_.head_num_ = head_num;
    args_.size_per_head_ = size_per_head;
    args_.hidden_units_ = head_num * size_per_head;
    args_.decoder_layers_ = decoder_layers;
    args_.vocab_size_ = vocab_size;
    args_.start_id_ = start_id;
    args_.end_id_ = end_id;
    args_.beam_search_diversity_rate_ = beam_search_diversity_rate;
    if (args_.beam_width_ > 16 || args_.beam_width_ > MAX_K)
      is_fuse_topk_softMax_ = false;
    if (std::is_same<DataType_, float>::value)
      args_.vocab_size_padded_ = vocab_size;
    else if (std::is_same<DataType_, half>::value)
      args_.vocab_size_padded_ = (int)(ceil(vocab_size / 8.)) * 8;

    args_.alpha_ = alpha;
    args_.normalization_before_ = normalization_before;
    args_.pos_offset_ = pos_offset;
    args_.pos_bias_ = pos_bias;
    args_.act_ = act;

    args_.prefix_lm_ = prefix_lm;
    args_.is_mbart_ = is_mbart;

    args_.finished_candidate_num_ = (finished_candidate_num == -1)
                                        ? beam_width * 2
                                        : finished_candidate_num;
    args_.early_stopping_ = early_stopping;

    K_cache_ = new DataType_ *[2];
    V_cache_ = new DataType_ *[2];

    K_mem_cache_ = new DataType_ *[args_.decoder_layers_];
    V_mem_cache_ = new DataType_ *[args_.decoder_layers_];

    decoder_ = new OpenDecoder<OpType_>(head_num,
                                        size_per_head,
                                        memory_hidden_units,
                                        is_fuse_qkv,
                                        normalization_before,
                                        args_.act_);
    decoder_->set_max_batch_size(batch_size * beam_width);

    size_t from_tensor_size =
        args_.batch_size_ * args_.beam_width_ * args_.hidden_units_;  // type T
    size_t decoder_workspace_size = decoder_->getWorkspaceSize();     // type T
    size_t decoder_normed_result_buffer_size =
        args_.batch_size_ * args_.beam_width_ * args_.hidden_units_;  // type T
    size_t cache_size = (prefix_lm)
                            ? (args_.batch_size_ * args_.beam_width_ *
                               (args_.seq_len_ + args_.memory_max_seq_len_) *
                               args_.hidden_units_)
                            : (args_.batch_size_ * args_.beam_width_ *
                               args_.seq_len_ * args_.hidden_units_);  // type T
    size_t mem_cache_size =
        (prefix_lm) ? 0 : (args_.batch_size_ * args_.beam_width_ *
                           memory_max_seq_len * args_.hidden_units_);  // type T

    size_t logits_buf_size = args_.batch_size_ * args_.beam_width_ *
                             args_.vocab_size_padded_;  // type float
    size_t cum_log_buf_size =
        args_.batch_size_ * args_.beam_width_;  // type float
    size_t word_ids_buf_size =
        args_.batch_size_ * args_.beam_width_;  // type int
    size_t parent_ids_buf_size =
        keep_alive_beam_ ? word_ids_buf_size : 0;  // type int
    size_t finished_buf_size =
        args_.batch_size_ * args_.beam_width_;  // type bool
    size_t alive_finished_buf_size = keep_alive_beam_ ? finished_buf_size : 0;
    size_t finished_count_size = (size_t)(ceil(1 / 32.)) * 32;  // type int

    size_t storage_size_per_beam =
        2 * args_.beam_width_ +
        SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * MAX_K + 2);
    args_.temp_storage_size_ = args_.batch_size_ * args_.beam_width_ *
                               storage_size_per_beam;  // type float
    args_.temp_storage_size_ = (size_t)(
        ceil(args_.batch_size_ * args_.beam_width_ * args_.beam_width_ / 4.) *
            4 * 2 +
        ceil(args_.batch_size_ * args_.beam_width_ *
             SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * MAX_K + 2) / 4.) *
            4);
    size_t padded_embedding_kernel_size =
        args_.hidden_units_ * args_.vocab_size_padded_;
    size_t padded_embedding_bias_size = args_.vocab_size_padded_;
    if (std::is_same<DataType_, float>::value ||
        (std::is_same<DataType_, half>::value &&
         args_.vocab_size_padded_ == args_.vocab_size_)) {
      padded_embedding_kernel_size = 0;
      padded_embedding_bias_size = 0;
    }

    // When using separated alive and finish beam queues, some buffers size need
    // to be doubled to restore beam search intermedia results of both alive and
    // finish beams.
    if (keep_alive_beam_ == true) {
      // cumulated log-probs of finish beams and alive beams
      cum_log_buf_size += cum_log_buf_size;
      finished_buf_size += finished_buf_size;
      // Double the size of topk_tmp_id_buf, topk_tmp_val_buf, since we need
      // select the top 2*beam_width.
      args_.temp_storage_size_ +=
          ceil(args_.batch_size_ * args_.beam_width_ * args_.beam_width_ / 4.) *
          4 * 2;
// Double tmp_buffer since we need select the top 2*beam_width.
#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
      args_.temp_storage_size_ +=
          ceil(args_.batch_size_ * args_.beam_width_ *
               SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * MAX_K) / 4.) *
          4;
#endif
    }

    // prevent memory misalinged address
    logits_buf_size = (size_t)(ceil(logits_buf_size / 4.)) * 4;
    cum_log_buf_size = (size_t)(ceil(cum_log_buf_size / 4.)) * 4;
    word_ids_buf_size = (size_t)(ceil(word_ids_buf_size / 4.)) * 4;
    parent_ids_buf_size = (size_t)(ceil(parent_ids_buf_size / 4.)) * 4;
    finished_buf_size = (size_t)(ceil(finished_buf_size / 32.)) * 32;
    alive_finished_buf_size =
        (size_t)(ceil(alive_finished_buf_size / 32.)) * 32;
    const size_t tmp_logits_buf_size = logits_buf_size;

    // get workspace size of topk kernel
    if (keep_alive_beam_ == true) {
      topK_update_kernelLauncher(topK_kernel_workspace,
                                 topk_workspace_size_,
                                 logits_buf_,
                                 finished_buf_,
                                 alive_finished_buf_,
                                 nullptr,
                                 word_ids_buf_,
                                 parent_ids_buf_,
                                 nullptr,
                                 nullptr,
                                 cum_log_buf_,
                                 0,
                                 args_,
                                 0);
    } else {
      topK_kernelLauncher(topK_kernel_workspace,
                          topk_workspace_size_,
                          logits_buf_,
                          word_ids_buf_,
                          finished_buf_,
                          args_,
                          0);
    }

    size_t lm_head_buffer_size = (prefix_lm)
                                     ? decoder_normed_result_buffer_size
                                     : decoder_normed_result_buffer_size * 3;

    size_t datatype_buf_size =
        from_tensor_size * 2 + decoder_workspace_size +
        (cache_size * 4 + mem_cache_size * 2) * args_.decoder_layers_ +
        lm_head_buffer_size;

    buf_ = reinterpret_cast<void *>(allocator_.malloc(
        ((sizeof(DataType_) == sizeof(half)) ? CUBLAS_WORKSPACE_SIZE : 0) +
        sizeof(DataType_) * datatype_buf_size +
        sizeof(float) * (logits_buf_size + cum_log_buf_size) +
        sizeof(DataType_) * tmp_logits_buf_size +
        sizeof(DataType_) * padded_embedding_kernel_size +
        sizeof(float) * padded_embedding_bias_size +
        sizeof(int) * (word_ids_buf_size + parent_ids_buf_size) +
        sizeof(bool) * (finished_buf_size + alive_finished_buf_size) +
        topk_workspace_size_ +
        sizeof(float) * args_.temp_storage_size_ +  // should be always float
        sizeof(int) * finished_count_size));

    if (sizeof(DataType_) == sizeof(half)) {
      cublas_workspace_ = buf_;
      from_tensor_[0] =
          (DataType_ *)((char *)cublas_workspace_ + CUBLAS_WORKSPACE_SIZE);
    } else {
      cublas_workspace_ = nullptr;
      from_tensor_[0] = (DataType_ *)(buf_);
    }
    from_tensor_[1] = (DataType_ *)(from_tensor_[0] + from_tensor_size);

    for (int i = 0; i < args_.decoder_layers_; ++i) {
      K_mem_cache_[i] =
          from_tensor_[1] + from_tensor_size + i * mem_cache_size * 2;
      V_mem_cache_[i] = from_tensor_[1] + from_tensor_size +
                        i * mem_cache_size * 2 + mem_cache_size;
    }
    if (args_.beam_width_ > 1) {
      /* We use two-way buffer since we have to update KV buf at the end of each
       * step. */
      K_cache_[0] = V_mem_cache_[decoder_layers - 1] + mem_cache_size +
                    0 * cache_size * args_.decoder_layers_;
      K_cache_[1] = V_mem_cache_[decoder_layers - 1] + mem_cache_size +
                    1 * cache_size * args_.decoder_layers_;
      V_cache_[0] = V_mem_cache_[decoder_layers - 1] + mem_cache_size +
                    2 * cache_size * args_.decoder_layers_;
      V_cache_[1] = V_mem_cache_[decoder_layers - 1] + mem_cache_size +
                    3 * cache_size * args_.decoder_layers_;
    } else {
      // if beam width is 1, we only need one buffer
      K_cache_[0] = V_mem_cache_[decoder_layers - 1] + mem_cache_size +
                    0 * cache_size * args_.decoder_layers_;
      K_cache_[1] = K_cache_[0];
      V_cache_[0] = V_mem_cache_[decoder_layers - 1] + mem_cache_size +
                    2 * cache_size * args_.decoder_layers_;
      V_cache_[1] = V_cache_[0];
    }

    decoder_buf_ = V_cache_[1] + cache_size * args_.decoder_layers_;

    if (prefix_lm) {
      trans_out_buf_ = (decoder_buf_ + decoder_workspace_size);
      lm_normed_result_buf_ =
          (trans_out_buf_ + decoder_normed_result_buffer_size);

      decoder_normed_result_buf_ =
          (lm_normed_result_buf_ + decoder_normed_result_buffer_size);
      // Used for post-norm.
      embedding_buf_ =
          (lm_normed_result_buf_ + decoder_normed_result_buffer_size);
    } else {
      decoder_normed_result_buf_ = (decoder_buf_ + decoder_workspace_size);
      // Used for post-norm.
      embedding_buf_ = (decoder_buf_ + decoder_workspace_size);
    }

    logits_buf_ = (float *)(decoder_normed_result_buf_ +
                            decoder_normed_result_buffer_size);
    cum_log_buf_ = (float *)(logits_buf_ + logits_buf_size);
    word_ids_buf_ = (int *)(cum_log_buf_ + cum_log_buf_size);
    parent_ids_buf_ = (int *)(word_ids_buf_ + word_ids_buf_size);
    finished_buf_ = (bool *)(parent_ids_buf_ + parent_ids_buf_size);
    alive_finished_buf_ = (bool *)(finished_buf_ + finished_buf_size);
    temp_storage_ = (float *)(alive_finished_buf_ + alive_finished_buf_size);
    finished_count_buf_ = (int *)(temp_storage_ + args_.temp_storage_size_);
    topK_kernel_workspace = (void *)(finished_count_buf_ + finished_count_size);
    padded_embedding_kernel =
        (DataType_ *)((char *)topK_kernel_workspace + topk_workspace_size_);
    padded_embedding_bias =
        (DataType_ *)(padded_embedding_kernel + padded_embedding_kernel_size);
    tmp_logits_buf_ =
        (DataType_ *)(padded_embedding_bias + padded_embedding_bias_size);

    h_finished_buf_ = new bool[finished_buf_size];
    h_trg_length_ = new int[args_.batch_size_];

    int isConfigExist = access("decoding_gemm_config.in", 0);
    if (isConfigExist == -1) {
      printf("[WARNING] decoding_gemm_config.in is not found\n");
    } else {
      readAlgoFromConfig(cublasAlgoMap_, 1);
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
  }

  void forward(const DecoderInitParam<DataType_> *param,
               DecodingInitParam<DataType_> decoding_params) {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    const int m = args_.batch_size_ * args_.beam_width_;
    const int k = args_.hidden_units_;
    const int n = args_.vocab_size_padded_;
    const DataType_ *embedding_kernel_ptr = nullptr;
    const DataType_ *embedding_bias_ptr = nullptr;

    int min_trg_len = 0;
    int max_trg_len = 0;

    if (decoding_params.trg_word) {
      cudaMemcpy(h_trg_length_,
                 decoding_params.trg_length,
                 sizeof(int) * args_.batch_size_,
                 cudaMemcpyDeviceToHost);
      min_trg_len = h_trg_length_[0];
      max_trg_len = h_trg_length_[0];

      for (int i = 1; i < args_.batch_size_; ++i) {
        min_trg_len = std::min(min_trg_len, h_trg_length_[i]);
        max_trg_len = std::max(max_trg_len, h_trg_length_[i]);
      }
    }

    /*
      sequence_length initialize to 0
      finished: false
      word_ids: start_id_
      cum_log_probs (for eacm beam, the first element is 0). e.g., [0 -inf -inf
      -inf][0 -inf -inf -inf]
      cum_log_probs: If keep_alive_beam_ is true, the first alive element is 0.
    */
    if (keep_alive_beam_ == true) {
      init_kernelLauncher_v2<float>(finished_buf_,
                                    alive_finished_buf_,
                                    decoding_params.sequence_length,
                                    word_ids_buf_,
                                    cum_log_buf_,
                                    args_.start_id_,
                                    args_.batch_size_,
                                    args_.beam_width_ * 2,
                                    decoding_params.stream);
    } else {
      init_kernelLauncher(finished_buf_,
                          decoding_params.sequence_length,
                          word_ids_buf_,
                          cum_log_buf_,
                          args_.start_id_,
                          args_.batch_size_,
                          args_.beam_width_,
                          decoding_params.stream);
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

/*
  User can check the init by init_kernel_check.
  init_kernel_check will compare the results of GPU and CPU.
  Note that init_kernel_check contains init and uses do not need to call it
  again.
*/
// init_kernel_check(finished_buf_, decoding_params.sequence_length,
// word_ids_buf_, cum_log_buf_,
//                   start_id_, batch_size_, beam_width_,
//                   decoding_params.stream);
#endif

    if (std::is_same<DataType_, float>::value ||
        (std::is_same<DataType_, half>::value &&
         args_.vocab_size_padded_ == args_.vocab_size_)) {
      embedding_kernel_ptr =
          (const DataType_ *)decoding_params.embedding_kernel;
      embedding_bias_ptr = (const DataType_ *)decoding_params.embedding_bias;
    } else if (std::is_same<DataType_, half>::value) {
      kernel_padding_kernelLauncher(padded_embedding_kernel,
                                    decoding_params.embedding_kernel,
                                    args_.hidden_units_,
                                    args_.vocab_size_,
                                    args_.vocab_size_padded_,
                                    decoding_params.stream);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      bias_padding_kernelLauncher(padded_embedding_bias,
                                  decoding_params.embedding_bias,
                                  args_.vocab_size_,
                                  args_.vocab_size_padded_,
                                  decoding_params.stream);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
      embedding_kernel_ptr = padded_embedding_kernel;
      embedding_bias_ptr = padded_embedding_bias;
    }

    int cache_size =
        (args_.prefix_lm_)
            ? (m * (args_.seq_len_ + args_.memory_max_seq_len_) *
               args_.hidden_units_)
            : (m * args_.seq_len_ * args_.hidden_units_);  // type T

    if (args_.prefix_lm_) {
      for (int layer = 0; layer < args_.decoder_layers_; ++layer) {
        // Use batch major
        // put k/v_buf from shape [B, H, L, Dh]
        // to cache [B, H, Dh/x, L, x]  and [B, H, L, Dh/x, x]
        transpose_cache_batch_major_kernelLauncher(
            K_cache_[1] + layer * cache_size,
            V_cache_[1] + layer * cache_size,
            param[layer].k_cache,
            param[layer].v_cache,
            decoding_params.memory_sequence_length,
            args_.batch_size_ * args_.beam_width_,
            args_.memory_max_seq_len_,
            args_.seq_len_ + args_.memory_max_seq_len_,
            args_.size_per_head_,
            args_.head_num_,
            decoding_params.stream);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
      }
    }

    for (uint step = 1; step <= args_.seq_len_; ++step) {
      // we use two-way buffer
      int kv_cache_id = step & 0x1;
      if (args_.normalization_before_) {
        if (args_.prefix_lm_) {
          embeddings_kernel_launcher(from_tensor_[0],
                                     decoding_params.embedding_table,
                                     decoding_params.position_encoding_table,
                                     decoding_params.type_table,
                                     decoding_params.memory_sequence_length,
                                     decoding_params.type_id,
                                     word_ids_buf_,
                                     step,
                                     m,
                                     args_.hidden_units_,
                                     args_.pos_bias_,
                                     decoding_params.stream);
        } else {
          if (args_.is_mbart_) {
            embedding_lookup_sine_position_encoding_kernel_launcher(
                embedding_buf_,
                decoding_params.embedding_table,
                decoding_params.position_encoding_table +
                    (step - 1 + args_.pos_offset_) * args_.hidden_units_,
                word_ids_buf_,
                m,
                args_.hidden_units_,
                decoding_params.stream);

            layer_norm(embedding_buf_,
                       decoding_params.mbart_layernorm.gamma,
                       decoding_params.mbart_layernorm.beta,
                       from_tensor_[0],
                       m,
                       k,
                       decoding_params.stream);

          } else {
            embedding_lookup_sine_position_encoding_kernel_launcher(
                from_tensor_[0],
                decoding_params.embedding_table,
                decoding_params.position_encoding_table +
                    (step - 1 + args_.pos_offset_) * args_.hidden_units_,
                word_ids_buf_,
                m,
                args_.hidden_units_,
                decoding_params.stream);
          }
        }
      } else {
        if (args_.prefix_lm_) {
          embeddings_kernel_launcher(embedding_buf_,
                                     decoding_params.embedding_table,
                                     decoding_params.position_encoding_table,
                                     decoding_params.type_table,
                                     decoding_params.memory_sequence_length,
                                     decoding_params.type_id,
                                     word_ids_buf_,
                                     step,
                                     m,
                                     args_.hidden_units_,
                                     args_.pos_bias_,
                                     decoding_params.stream);
        } else {
          // TODO(gongenlei): Only support Bart temporarily.
          embedding_position_lookups_bart_kernel_launcher(
              embedding_buf_,
              decoding_params.embedding_table,
              decoding_params.position_encoding_table +
                  (step - 1 + args_.pos_offset_) * args_.hidden_units_,
              word_ids_buf_,
              m,
              args_.hidden_units_,
              decoding_params.stream);
        }

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        layer_norm(embedding_buf_,
                   decoding_params.layernorm.gamma,
                   decoding_params.layernorm.beta,
                   from_tensor_[0],
                   m,
                   k,
                   decoding_params.stream);
      }

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      int from_id, out_id;
      for (int layer = 0; layer < args_.decoder_layers_; ++layer) {
        /*
          For the first layer (layer-0), from_id is 0. We also stored the
          embedding lookup
          result in from_tensor_[0]
        */
        from_id = layer & 0x1;
        out_id = 1 - from_id;

        /*
          We use one decoder_ object to process multiple decoder layers.

          At the beginning of each decoder layer, we initialize the decoder
          object
          with corresponding weights and decoder_buf_.
          The decoder_buf_ is reused.
        */
        decoder_->initialize(param[layer], decoder_buf_, cublas_workspace_);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        if (args_.prefix_lm_) {
          decoder_->forward_v2(
              from_tensor_[from_id],
              nullptr,
              K_cache_[kv_cache_id] + layer * cache_size,
              V_cache_[kv_cache_id] + layer * cache_size,
              nullptr,
              nullptr,
              nullptr,
              from_tensor_[out_id],
              step + args_.memory_max_seq_len_,
              args_.seq_len_ + args_.memory_max_seq_len_,
              false, /* is_cross_attention */
              keep_alive_beam_ ? alive_finished_buf_ : finished_buf_,
              args_.memory_max_seq_len_,
              decoding_params.memory_sequence_length);
        } else {
          decoder_->forward(
              from_tensor_[from_id],
              decoding_params.memory_tensor,
              K_cache_[kv_cache_id] + layer * cache_size,
              V_cache_[kv_cache_id] + layer * cache_size,
              K_mem_cache_[layer],
              V_mem_cache_[layer],
              decoding_params.memory_sequence_length,
              from_tensor_[out_id],
              step,
              args_.seq_len_,
              true, /* is_cross_attention */
              keep_alive_beam_ ? alive_finished_buf_ : finished_buf_);
        }

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
      }

      if (step > min_trg_len) {
        DataType_ alpha = (DataType_)1.0f;
        DataType_ beta = (DataType_)0.0f;

        if (args_.prefix_lm_) {
          if (args_.normalization_before_) {
            layer_norm(from_tensor_[out_id],
                       decoding_params.layernorm.gamma,
                       decoding_params.layernorm.beta,
                       decoder_normed_result_buf_,
                       m,
                       k,
                       decoding_params.stream);

#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif

            // trans here
            cublasMM_cublasLtMM_wrapper_decoder(decoding_params.cublaslt_handle,
                                                decoding_params.cublas_handle,
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                k,
                                                m,
                                                k,
                                                &alpha,
                                                decoding_params.trans_kernel,
                                                AType_,
                                                k,
                                                decoder_normed_result_buf_,
                                                BType_,
                                                k,
                                                &beta,
                                                trans_out_buf_,
                                                CType_,
                                                k,
                                                decoding_params.stream,
                                                cublasAlgoMap_,
                                                cublas_workspace_);
          } else {
            // trans here
            cublasMM_cublasLtMM_wrapper_decoder(decoding_params.cublaslt_handle,
                                                decoding_params.cublas_handle,
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                k,
                                                m,
                                                k,
                                                &alpha,
                                                decoding_params.trans_kernel,
                                                AType_,
                                                k,
                                                from_tensor_[out_id],
                                                BType_,
                                                k,
                                                &beta,
                                                trans_out_buf_,
                                                CType_,
                                                k,
                                                decoding_params.stream,
                                                cublasAlgoMap_,
                                                cublas_workspace_);
          }
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif

          // add bias decoding_params.trans_bias
          add_bias_act_kernelLauncher(trans_out_buf_,
                                      decoding_params.trans_bias,
                                      m,
                                      k,
                                      args_.act_,
                                      decoding_params.stream);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif

          layer_norm(trans_out_buf_,
                     decoding_params.lm_layernorm.gamma,
                     decoding_params.lm_layernorm.beta,
                     lm_normed_result_buf_,
                     m,
                     k,
                     decoding_params.stream);

#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif

          cublasMM_cublasLtMM_wrapper_decoder(decoding_params.cublaslt_handle,
                                              decoding_params.cublas_handle,
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_N,
                                              n,
                                              m,
                                              k,
                                              &alpha,
                                              embedding_kernel_ptr,
                                              AType_,
                                              n,
                                              lm_normed_result_buf_,
                                              BType_,
                                              k,
                                              &beta,
                                              tmp_logits_buf_,
                                              CType_,
                                              n,
                                              decoding_params.stream,
                                              cublasAlgoMap_,
                                              cublas_workspace_);

        } else {
          if (args_.normalization_before_) {
            layer_norm(from_tensor_[out_id],
                       decoding_params.layernorm.gamma,
                       decoding_params.layernorm.beta,
                       decoder_normed_result_buf_,
                       m,
                       k,
                       decoding_params.stream);

#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif

            cublasMM_cublasLtMM_wrapper_decoder(decoding_params.cublaslt_handle,
                                                decoding_params.cublas_handle,
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                n,
                                                m,
                                                k,
                                                &alpha,
                                                embedding_kernel_ptr,
                                                AType_,
                                                n,
                                                decoder_normed_result_buf_,
                                                BType_,
                                                k,
                                                &beta,
                                                tmp_logits_buf_,
                                                CType_,
                                                n,
                                                decoding_params.stream,
                                                cublasAlgoMap_,
                                                cublas_workspace_);

          } else {
            // Post-norm
            cublasMM_cublasLtMM_wrapper_decoder(decoding_params.cublaslt_handle,
                                                decoding_params.cublas_handle,
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                n,
                                                m,
                                                k,
                                                &alpha,
                                                embedding_kernel_ptr,
                                                AType_,
                                                n,
                                                from_tensor_[out_id],
                                                BType_,
                                                k,
                                                &beta,
                                                tmp_logits_buf_,
                                                CType_,
                                                n,
                                                decoding_params.stream,
                                                cublasAlgoMap_,
                                                cublas_workspace_);
          }
        }

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        if (decoding_params.logits_mask) {
          apply_logits_mask_kernelLauncher(
              tmp_logits_buf_,
              keep_alive_beam_ ? alive_finished_buf_ : finished_buf_,
              args_.batch_size_,
              args_.beam_width_,
              args_.vocab_size_padded_,
              args_.vocab_size_,
              decoding_params.stream,
              decoding_params.logits_mask);
#ifndef NDEBUG
          cudaDeviceSynchronize();
          check_cuda_error(cudaGetLastError());
#endif
        }

        // Beamsearch
        if (is_fuse_topk_softMax_) {
          if (keep_alive_beam_) {
            // Use separated alive and finish beam queues to avoid the decrease
            // of alive beams.
            topK_softMax_update(tmp_logits_buf_,
                                embedding_bias_ptr,
                                finished_buf_,
                                alive_finished_buf_,
                                decoding_params.sequence_length,
                                word_ids_buf_,
                                parent_ids_buf_,
                                decoding_params.output_ids + (step - 1) * m * 2,
                                decoding_params.parent_ids + (step - 1) * m * 2,
                                cum_log_buf_,
                                reinterpret_cast<void *>(temp_storage_),
                                step,
                                args_,
                                decoding_params.stream);
          } else {
            topK_softMax(tmp_logits_buf_,
                         embedding_bias_ptr,
                         finished_buf_,
                         cum_log_buf_,
                         word_ids_buf_,
                         reinterpret_cast<void *>(temp_storage_),
                         args_,
                         decoding_params.stream);
#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif

            update_kernelLauncher_v2(
                finished_buf_,
                decoding_params.parent_ids + (step - 1) * m,
                decoding_params.sequence_length,
                word_ids_buf_,
                decoding_params.output_ids + (step - 1) * m,
                finished_count_buf_,
                args_,
                decoding_params.stream);
#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
          }

        } else {
          if (keep_alive_beam_ == true) {
            update_logits_v2(tmp_logits_buf_,
                             embedding_bias_ptr,
                             args_.end_id_,
                             finished_buf_,
                             m,
                             n,
                             decoding_params.stream);

            // Use separated alive and finish beam queues to avoid the decrease
            // of alive beams.
            topK_update_kernelLauncher(
                topK_kernel_workspace,
                topk_workspace_size_,
                tmp_logits_buf_,
                finished_buf_,
                alive_finished_buf_,
                decoding_params.sequence_length,
                word_ids_buf_,
                parent_ids_buf_,
                decoding_params.output_ids + (step - 1) * m * 2,
                decoding_params.parent_ids + (step - 1) * m * 2,
                cum_log_buf_,
                step,
                args_,
                decoding_params.stream);
          } else {
            update_logits(logits_buf_,
                          tmp_logits_buf_,
                          embedding_bias_ptr,
                          args_.end_id_,
                          finished_buf_,
                          m,
                          n,
                          decoding_params.stream);

#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());

/*
  User can check the update_logits by update_logits_kernel_check.
  update_logits_kernel_check will compare the results of GPU and CPU.
  Note that update_logits_kernel_check contains update_logits and uses do not
  need to call it again.
*/
// update_logits_kernel_check(logits_buf_, decoding_params.embedding_bias,
// args_.end_id_, finished_buf_, m, n, decoding_params.stream);
#endif
            /* adding cum_log_buf_ to logits_buf_ */
            broadcast_kernelLauncher(logits_buf_,
                                     cum_log_buf_,
                                     args_.batch_size_,
                                     args_.beam_width_,
                                     args_.vocab_size_padded_,
                                     decoding_params.stream);
#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());

/*
  User can check the broadcast_kernel by broadcast_kernel_check.
  broadcast_kernel_check will compare the results of GPU and CPU.
  Note that broadcast_kernel_check contains broadcast_kernelLauncher and uses do
  not need to call it again.
*/
// broadcast_kernel_check(logits_buf_, cum_log_buf_, batch_size_, beam_width_,
// vocab_size_, decoding_params.stream);
#endif

            topK_kernelLauncher(topK_kernel_workspace,
                                topk_workspace_size_,
                                logits_buf_,
                                word_ids_buf_,
                                finished_buf_,
                                args_,
                                decoding_params.stream);
#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
            update_kernelLauncher(logits_buf_,
                                  cum_log_buf_,
                                  finished_buf_,
                                  decoding_params.parent_ids + (step - 1) * m,
                                  decoding_params.sequence_length,
                                  word_ids_buf_,
                                  decoding_params.output_ids + (step - 1) * m,
                                  args_.batch_size_,
                                  args_.beam_width_,
                                  args_.vocab_size_padded_,
                                  decoding_params.stream,
                                  args_.end_id_,
                                  finished_count_buf_);
          }
        }

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
      }

      if (step <= max_trg_len) {
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        update_with_force_decodingLauncher<float>(
            decoding_params.trg_word,
            decoding_params.trg_length,
            finished_buf_,
            word_ids_buf_,
            (step > min_trg_len) ? nullptr : decoding_params.sequence_length,
            (keep_alive_beam_) ? parent_ids_buf_ : nullptr,
            (keep_alive_beam_) ? decoding_params.parent_ids + (step - 1) * m * 2
                               : decoding_params.parent_ids + (step - 1) * m,
            (keep_alive_beam_) ? decoding_params.output_ids + (step - 1) * m * 2
                               : decoding_params.output_ids + (step - 1) * m,
            cum_log_buf_,
            keep_alive_beam_,
            args_.batch_size_,
            (keep_alive_beam_) ? args_.beam_width_ * 2 : args_.beam_width_,
            max_trg_len,
            step,
            decoding_params.stream);
      }

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      if (args_.beam_width_ > 1) {
        // chose which self cache to use
        int decoder_max_seq_len =
            (decoder_->getCacheFormat() != 0) ? args_.seq_len_ : -1;

        update_KV_cache_kernelLauncher_v2(
            K_cache_,
            V_cache_,
            keep_alive_beam_ ? parent_ids_buf_
                             : decoding_params.parent_ids + (step - 1) * m,
            keep_alive_beam_ ? alive_finished_buf_ : finished_buf_,
            args_.batch_size_,
            args_.beam_width_,
            args_.head_num_,
            args_.size_per_head_,
            step,
            decoder_max_seq_len,
            cache_size,
            args_.decoder_layers_,
            decoding_params.stream,
            (args_.prefix_lm_) ? args_.memory_max_seq_len_ : -1);
      }
#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());

/*
  User can check the update_KV_cache by update_KV_cache_kernel_check.
  update_KV_cache_kernel_check will compare the results of GPU and CPU.
  Note that update_KV_cache_kernel_check contains update_KV_cache and uses do
  not need to call it again.
*/
// update_KV_cache_kernel_check(K_cache_, V_cache_, decoding_params.parent_ids +
// (step - 1) * batch_size_ * beam_width_, batch_size_, beam_width_,
// hidden_units_, step, cache_size, decoder_layers_, decoding_params.stream);
#endif

      if (step > max_trg_len) {
        // TODO Find a better method to check the is_finished
        int finish_size = (keep_alive_beam_) ? m * 2 : m;
        cudaMemcpy(h_finished_buf_,
                   finished_buf_,
                   sizeof(bool) * finish_size,
                   cudaMemcpyDeviceToHost);
        int sum = 0;
        for (int i = 0; i < finish_size; i++) {
          sum += (int)h_finished_buf_[i];
        }
        if (sum == finish_size) break;
      }
    }  // end for decoding step for llop
  }    // end of forward

  virtual ~DecodingBeamsearch() {
    delete[] K_cache_;
    delete[] V_cache_;
    delete[] K_mem_cache_;
    delete[] V_mem_cache_;
    delete[] h_finished_buf_;
    delete[] h_trg_length_;
    delete decoder_;
    allocator_.free(buf_);
  }
};

}  // namespace fastertransformer
