/*
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
 * BERT Encoder transformer
 **/

#pragma once

#include <cuda_runtime.h>
#include "fastertransformer/cuda/cuda_int8_kernels.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/cuda/open_attention.h"
#include "fastertransformer/gemm_test/encoder_gemm_func.h"
#include "fastertransformer/gemm_test/encoder_igemm_func.h"
#include "fastertransformer/utils/allocator.h"
#include "fastertransformer/utils/common_structure.h"
#include "fastertransformer/utils/functions.h"

namespace fastertransformer {

template <typename T>
class BertInitParam {
public:
  const T *from_tensor = nullptr;
  const T *to_tensor = nullptr;

  AttentionWeight<T> self_attention;
  const T *attr_mask = nullptr;
  LayerNormWeight<T> self_layernorm;

  FFNWeight<T> ffn;
  LayerNormWeight<T> ffn_layernorm;

  T *transformer_out;
  cublasHandle_t cublas_handle = nullptr;
  cublasLtHandle_t cublaslt_handle = nullptr;
  cudaStream_t stream = 0;

  const int *sequence_id_offset = nullptr;
  int valid_word_num = -1;
  int layer_idx = 0;
  int layer_num = 12;

  // Part 1:
  //  First 80 are for activation amaxs. For each activation amax, there are 4
  //  values: amax, amax/127.0f, amax/127.0f/127.0f, 127.0f/amax -- input_amax
  //  0-3 , Q_aftergemm_amax 4-7, Qbias_amax 8-11, K_aftergemm_amax 12-15,
  //  Kbias_amax 16-19, V_aftergemm_amax 20-23, Vbias_amax 24-27, bmm1_amax
  //  28-31, Softmax_amax 32-35, bmm2_amax 36-39, Proj_aftergemm_scale 40-43,
  //  ProjBiasNorm_amax 44-47, FC1_aftergemm_amax 48-51, F1Bias_amax 52-55,
  //  FC2_aftergemm_amax 56-59, F2BiasNorm_amax 60-63, reserve 64-79
  // Part 2:
  //  Kernel amaxs, for each kernel amax list, there are output_channel values :
  //  query_weight_amax_list, key_weight_amax_list, value_weight_amax_list,
  //  proj_weight_amax_list, FC1_weight_amax_list, FC2_weight_amax_list
  // Part 3:
  //  Int8 gemm deQFactor list (8 values): Q_deQ_scale, K_deQ_scale,
  //  V_deQ_scale, bmm1_deQ_scale, bmm2_deQ_scale, FC0_deQ_scale, FC1_deQ_scale,
  //  FC2_deQ_scale
  // Part 4:
  //  Amax used in trt fused mha kernel (3 values) : QKVbias_amax, Softmax_amax,
  //  bmm2_amax
  const float *amaxList = nullptr;
  const int *trt_seqlen_offset = nullptr;
  int trt_seqlen_size = -1;
};

template <OperationType OpType_,
          template <OperationType> class MultiHeadAttention_>
class BertEncoderTransformerTraits;

template <template <OperationType> class MultiHeadAttention_>
class BertEncoderTransformerTraits<OperationType::FP32, MultiHeadAttention_>
    : public TransformerTraits<OperationType::FP32> {
public:
  typedef MultiHeadAttention_<OpType> MultiHeadAttention;
};

template <template <OperationType> class MultiHeadAttention_>
class BertEncoderTransformerTraits<OperationType::FP16, MultiHeadAttention_>
    : public TransformerTraits<OperationType::FP16> {
public:
  typedef MultiHeadAttention_<OpType> MultiHeadAttention;
};

template <class Traits_>
class BertEncoderTransformer {
  IAllocator *allocator_ = NULL;
  typename Traits_::MultiHeadAttention *attention_ = NULL;
  typedef typename Traits_::DataType DataType_;
  BertInitParam<DataType_> param_;

  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  std::map<std::string, cublasLtMatmulAlgo_info> cublasAlgoMap_;
  std::map<std::string, int> parameterMap_;

  DataType_ *buf_ = NULL;
  DataType_ *attr_out_buf_;
  DataType_ *attr_matmul_buf_;
  DataType_ *inter_matmul_buf_;
  DataType_ *attr_matmul_unnormed_buf_;
  void *cublas_workspace_ = NULL;

  int batch_size_;
  int from_seq_len_;
  int to_seq_len_;
  int head_num_;
  int size_per_head_;

  int sm_;
  bool allow_gemm_test_ = false;
  bool use_gelu_ = true;
  bool use_ORDER_COL32_2R_4R4_ = false;

  // for int8 quantization
  const float *FC0_weight_amax_list, *FC1_weight_amax_list,
      *FC2_weight_amax_list;
  float scale_list[INT8O_GEMM_NUM + TRT_FUSED_MHA_AMAX_NUM];
  const float *bmm2_amax_ptr, *ProjBiasNorm_amax_ptr, *F1Bias_amax_ptr,
      *F2BiasNorm_amax_ptr, *to_tensor_amax_ptr, *Proj_aftergemm_amax_ptr,
      *F1_aftergemm_amax_ptr, *F2_aftergemm_amax_ptr,
      *int8O_gemm_deQ_scale_list;
  // int8_mode == 0 -- not use int8
  // int8_mode == 1 -- use int8; without quantized residual; when (batch*seqLen
  // >= 512) or (seqLen % 32 !=0 ), using trt fused mha
  // int8_mode == 2 -- use int8; with quantized residual; with trt fused mha
  // int8_mode == 3 -- use int8; with quantized residual; without trt fused mha
  int int8_mode_;
  int layer_idx_;
  int layer_num_;
  const int8_t *int8_from_tensor_;
  const DataType_ *transA_from_tensor_;
  int32_t *int_buf_;
  DataType_ *tmp_DataType_, *transA_from_tensor_tmp_,
      *transformer_out_tmp_DataType_;
  int8_t *tmp_int8_, *int8_from_tensor_tmp_, *attr_matmul_buf_tmp_,
      *transformer_out_tmp_int8_;

public:
  void setLayerIdx(int layer_idx) { layer_idx_ = layer_idx; }

  size_t calBufSizeInByte(int batch_size,
                          int seq_len,
                          int head_num,
                          int size_per_head,
                          int int8_mode) {
    size_t m = batch_size * seq_len;
    size_t n = head_num * size_per_head;
    size_t k = n;
    size_t normal_buf_size;
    if (int8_mode != 0) {
      // transA_from_tensor & transformer_out_tmp_DataType
      normal_buf_size =
          m * k * sizeof(DataType_) +
          // int8_from_tensor & attr_matmul_buf_tmp & transformer_out_tmp_int8
          m * k * sizeof(int8_t) +
          // int8 qkv weight
          3 * n * k * sizeof(int8_t) +
          // FC0 & FC1 & FC2 for m*k(4k)*sizeof(DataType)
          4 * m * k * sizeof(int) +
          // attr_out_buf_ & attr_matmul_buf_ & inter_matmul_buf_
          6 * m * n * sizeof(DataType_) +
          // temp buf
          m * n * sizeof(DataType_);
    } else {
      normal_buf_size =
          sizeof(DataType_) * (m * n) * 7 +
          ((sizeof(half) == sizeof(DataType_)) ? CUBLAS_WORKSPACE_SIZE : 0);
    }
    return normal_buf_size;
  }

  bool checkParameterInMap(int batch_size,
                           int seq_len,
                           int head_num,
                           int size_per_head,
                           int int8_mode,
                           int is_fp16) {
    char mark[1000];
    bool parameterInMap;
    int dataType = is_fp16 == 0 ? FLOAT_DATATYPE : HALF_DATATYPE;
    if (int8_mode != 0) {
      dataType = INT8_DATATYPE;
    }
    sprintf(mark,
            "%d_%d_%d_%d_%d",
            batch_size,
            seq_len,
            head_num,
            size_per_head,
            dataType);
    if (parameterMap_.find(std::string(mark)) != parameterMap_.end())
      parameterInMap = true;
    else
      parameterInMap = false;
    return parameterInMap;
  }

  // free buffer for gemm test
  // This function requires the same allocator of allocateBufferForGemmTest(*)
  void freeBufferForGemmTest(IAllocator *allocator, void *&buffer) {
    if (buffer != NULL) {
      allocator->free(buffer);
      buffer = NULL;
    }
  }

  void allocateBufferForGemmTest(IAllocator *allocator,
                                 void *&buffer,
                                 int batch_size,
                                 int seq_len,
                                 int head_num,
                                 int size_per_head,
                                 int int8_mode,
                                 int is_fp16) {
    size_t buf_size_in_byte = calGemmTestBufSizeInByte(
        batch_size, seq_len, head_num, size_per_head, int8_mode, is_fp16);
    size_t total, free;
    check_cuda_error(cudaMemGetInfo(&free, &total));
    if (free < buf_size_in_byte + 10 * 1024 * 1024) {
      printf(
          "[WARNING] There is no enough device memory for gemm test!\n %ld "
          "Bytes is needed, but only %ld Bytes is free.\n",
          buf_size_in_byte,
          free);
      buffer = NULL;
      return;
    }
    buffer =
        reinterpret_cast<void *>(allocator->malloc(buf_size_in_byte, false));
  }

  bool gemmTest(int batch_size,
                int seq_len,
                int head_num,
                int size_per_head,
                int int8_mode,
                int is_fp16) {
    bool hasChangedConfig = false;
    if (int8_mode != 0) {
      // if not found parameters in map,
      // read config first
      // in case multiple instances (for example in tensorflow op) are used
      if (!checkParameterInMap(batch_size,
                               seq_len,
                               head_num,
                               size_per_head,
                               int8_mode,
                               is_fp16)) {
        readAlgoFromConfig(int8_mode, cublasAlgoMap_, parameterMap_);
      } else {
        return hasChangedConfig;
      }

      // if still not found algos in map,
      // do gemm test
      if (!checkParameterInMap(batch_size,
                               seq_len,
                               head_num,
                               size_per_head,
                               int8_mode,
                               is_fp16)) {
        void *gemm_test_buf = NULL;
        allocateBufferForGemmTest(allocator_,
                                  gemm_test_buf,
                                  batch_size,
                                  seq_len,
                                  head_num,
                                  size_per_head,
                                  int8_mode,
                                  is_fp16);
        if (gemm_test_buf != NULL) {
          generate_encoder_igemm_config(
              batch_size, seq_len, head_num, size_per_head, gemm_test_buf);
          freeBufferForGemmTest(allocator_, gemm_test_buf);
          readAlgoFromConfig(int8_mode, cublasAlgoMap_, parameterMap_);
          hasChangedConfig = true;
        }
      } else {
        hasChangedConfig = true;
        return hasChangedConfig;
      }
    } else {
      // if not found parameters in map,
      // read config first
      // in case multiple instances (for example in tensorflow op) are used
      if (!checkParameterInMap(batch_size,
                               seq_len,
                               head_num,
                               size_per_head,
                               int8_mode,
                               is_fp16)) {
        readAlgoFromConfig(int8_mode, cublasAlgoMap_, parameterMap_);
      } else {
        return hasChangedConfig;
      }

      // if still not found parameters in map,
      // do gemm test
      if (!checkParameterInMap(batch_size,
                               seq_len,
                               head_num,
                               size_per_head,
                               int8_mode,
                               is_fp16)) {
        void *gemm_test_buf = NULL;
        allocateBufferForGemmTest(allocator_,
                                  gemm_test_buf,
                                  batch_size,
                                  seq_len,
                                  head_num,
                                  size_per_head,
                                  int8_mode,
                                  is_fp16);
        if (gemm_test_buf != NULL) {
          if (is_fp16 == 1)
            generate_encoder_gemm_config<half>(
                batch_size, seq_len, head_num, size_per_head, gemm_test_buf);
          else
            generate_encoder_gemm_config<float>(
                batch_size, seq_len, head_num, size_per_head, gemm_test_buf);
          freeBufferForGemmTest(allocator_, gemm_test_buf);
          readAlgoFromConfig(int8_mode, cublasAlgoMap_, parameterMap_);
          hasChangedConfig = true;
        }
      } else {
        hasChangedConfig = true;
        return hasChangedConfig;
      }
    }
    return hasChangedConfig;
  }

  // free buffer for BertEncoderTransformer
  void freeBuffer() {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    if (buf_ != NULL) {
      if (allocator_ == NULL) {
        printf(
            "[ERROR][BertEncoderTransformer][freeBuffer] allocator_ is "
            "NULL!\n");
        exit(-1);
      }
      allocator_->free(buf_);
      buf_ = NULL;
    }
    if (attention_ != NULL) attention_->freeBuffer();
  }

  // allocate buffer for BertEncoderTransformer
  // do gemm test if allow_gemm_test == true
  void allocateBuffer(IAllocator *allocator,
                      int batch_size,
                      int from_seq_len,
                      int to_seq_len,
                      int head_num,
                      int size_per_head,
                      bool use_trt_kernel = true) {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    try {
      if (allocator == NULL) {
        printf(
            "[ERROR][BertEncoderTransformer][allocateBuffer] allocator == "
            "NULL!\n");
        exit(-1);
      }
      // only allocate new buffer when buf_ is empty
      // if buf_ is not empty, use previous allocated one
      // this can ensure consistency between (allocator_, batch_size_, ...) and
      // buf_
      if (buf_ != nullptr) {
        printf(
            "[ERROR][BertEncoderTransformer][allocateBuffer] previous buffer "
            "is not freed, use previous one. To allocate new buffer, please "
            "use freeBuffer() to free previous buffer first.\n");
        exit(-1);
      } else {
        allocator_ = allocator;
        batch_size_ = batch_size;
        from_seq_len_ = from_seq_len;
        to_seq_len_ = to_seq_len;
        head_num_ = head_num;
        size_per_head_ = size_per_head;

        int m = batch_size_ * from_seq_len_;
        int k = head_num_ * size_per_head_;
        int n = k;

        int buf_size = m * n;
        size_t buf_size_in_byte = calBufSizeInByte(
            batch_size_, from_seq_len_, head_num_, size_per_head_, int8_mode_);

        // allocate buffer
        if (int8_mode_ != 0) {
          buf_ = reinterpret_cast<DataType_ *>(
              allocator_->malloc(buf_size_in_byte, false));
          if (buf_ == nullptr)
            throw std::runtime_error(
                std::string("Allocator failed to allocate internal buffer."));

          attr_out_buf_ =
              (DataType_ *)(((char *)buf_) + m * k * sizeof(DataType_) +
                            m * k * sizeof(int8_t) +
                            3 * n * k * sizeof(int8_t) +
                            4 * m * k * sizeof(int));
          attr_matmul_buf_ = attr_out_buf_ + buf_size;
          inter_matmul_buf_ = attr_matmul_buf_ + buf_size;

          int8_from_tensor_tmp_ =
              (int8_t *)(((char *)buf_) + m * k * (sizeof(DataType_)));
          attr_matmul_buf_tmp_ = int8_from_tensor_tmp_;
          transformer_out_tmp_int8_ = int8_from_tensor_tmp_;
          transA_from_tensor_tmp_ = (DataType_ *)buf_;
          transformer_out_tmp_DataType_ = transA_from_tensor_tmp_;

          int_buf_ =
              (int32_t *)(((char *)buf_) +
                          (m * k) * (sizeof(DataType_) + sizeof(int8_t)) +
                          3 * n * k * sizeof(int8_t));

          tmp_DataType_ =
              (DataType_ *)(((char *)buf_) +
                            m * k * (sizeof(DataType_) + sizeof(int8_t)) +
                            3 * n * k * sizeof(int8_t) +
                            4 * m * k * sizeof(int32_t) +
                            6 * m * n * sizeof(DataType_));
          tmp_int8_ = (int8_t *)tmp_DataType_;
        } else {
          buf_ = reinterpret_cast<DataType_ *>(
              allocator_->malloc(buf_size_in_byte, false));
          if (buf_ == nullptr)
            throw std::runtime_error(
                std::string("Allocator failed to allocate internal buffer."));

          if (sizeof(half) == sizeof(DataType_)) {
            // cublas_workspace_ should be the start pointer of cudaMalloc()
            // to ensure 16B alignemnet
            cublas_workspace_ = buf_;
            attr_out_buf_ = (DataType_ *)((char *)cublas_workspace_ +
                                          CUBLAS_WORKSPACE_SIZE);
          } else {
            cublas_workspace_ = nullptr;
            attr_out_buf_ = (DataType_ *)buf_;
          }
          attr_matmul_buf_ = attr_out_buf_ + buf_size;
          inter_matmul_buf_ = attr_matmul_buf_ + buf_size;
          attr_matmul_unnormed_buf_ = inter_matmul_buf_ + 4 * buf_size;
        }
      }

      bool hasChangedConfig = false;
      int is_fp16;
      if (Traits_::OpType == OperationType::FP32)
        is_fp16 = 0;
      else
        is_fp16 = 1;
      // check if target algos in map
      if (allow_gemm_test_) {
        hasChangedConfig = gemmTest(batch_size_,
                                    from_seq_len_,
                                    head_num_,
                                    size_per_head_,
                                    int8_mode_,
                                    is_fp16);
      }

      // allocate buffer for attention_
      attention_->allocateBuffer(allocator,
                                 cublas_workspace_,
                                 batch_size_,
                                 from_seq_len_,
                                 to_seq_len,
                                 head_num_,
                                 size_per_head_,
                                 hasChangedConfig,
                                 use_trt_kernel);
    } catch (std::runtime_error &error) {
      throw error;
    }
  }

  BertEncoderTransformer(int int8_mode = 0,
                         bool allow_gemm_test = false,
                         bool use_gelu = true)
      : int8_mode_(int8_mode),
        allow_gemm_test_(allow_gemm_test),
        use_gelu_(use_gelu) {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif

    try {
      // sm_ = getSMVersion();
      // Set fake sm_ which have no effect.
      sm_ = 70;

      if (sm_ >= 80) {
        use_ORDER_COL32_2R_4R4_ = true;
      }
      if (sm_ < 75 && int8_mode_ != 0) {
        printf(
            "[ERROR][BertEncoderTransformer] int8 mode only works with sm >= "
            "75.\n");
        exit(-1);
      }

      int isConfigExist = -1;
      if (int8_mode_ != 0)
        isConfigExist = access(IGEMM_CONFIG, 0);
      else
        isConfigExist = access(GEMM_CONFIG, 0);
      if (isConfigExist == -1) {
        if (!allow_gemm_test_) {
          // printf(
          //     "[WARNING][BertEncoderTransformer] %s is not found; using "
          //     "default GEMM algo\n",
          //     int8_mode_ != 0 ? IGEMM_CONFIG : GEMM_CONFIG);
        }
      } else {
        readAlgoFromConfig(int8_mode_, cublasAlgoMap_, parameterMap_);
      }

      attention_ = new typename Traits_::MultiHeadAttention(
          int8_mode_, allow_gemm_test_, use_ORDER_COL32_2R_4R4_, sm_);
    } catch (std::runtime_error &error) {
      throw error;
    }
  }

  BertEncoderTransformer(const BertEncoderTransformer *transformer) {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    sm_ = transformer->sm_;
    use_ORDER_COL32_2R_4R4_ = transformer->use_ORDER_COL32_2R_4R4_;
    int8_mode_ = transformer->int8_mode_;
    allow_gemm_test_ = transformer->allow_gemm_test_;
    use_gelu_ = transformer->use_gelu_;

    cublasAlgoMap_ = transformer->cublasAlgoMap_;
    parameterMap_ = transformer->parameterMap_;

    attention_ =
        new typename Traits_::MultiHeadAttention(transformer->attention_);
  }

  void genTransATensorAndInt8TensorForFirstLayer() {
    const int m = param_.sequence_id_offset == nullptr
                      ? batch_size_ * from_seq_len_
                      : param_.valid_word_num;
    const int k = head_num_ * size_per_head_;
    if (int8_mode_ == 1) {
      transposeMatrix_colMajorToCOL32_kernelLauncher(
          transA_from_tensor_tmp_, param_.from_tensor, k, m, param_.stream);
      transA_from_tensor_ = (const DataType_ *)transA_from_tensor_tmp_;
      quantized_kernelLauncher(int8_from_tensor_tmp_,
                               transA_from_tensor_,
                               m * k,
                               to_tensor_amax_ptr + 3,
                               param_.stream);
    } else if (int8_mode_ == 2 || int8_mode_ == 3) {
      transposeMatrix_colMajorToCOL32_quantize_kernelLauncher(
          int8_from_tensor_tmp_,
          param_.from_tensor,
          k,
          m,
          to_tensor_amax_ptr + 3,
          param_.stream);
    }
    int8_from_tensor_ = (const int8_t *)(int8_from_tensor_tmp_);
  }

  /**
   * Initialize the parameters in class
   * We will keep the Ctor empty to ensure the sub classes follow the same init
   *routine.
   * Please be aware that no dynamic memory allocation should be placed
   **/
  void initialize(BertInitParam<DataType_> param) {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif

    param_ = param;
    cuda::MultiHeadInitParam<DataType_> multi_head_init_param;

    if (int8_mode_ != 0) {
      int hidden_dim = size_per_head_ * head_num_;
      layer_idx_ = param_.layer_idx;
      layer_num_ = param_.layer_num;

      bmm2_amax_ptr = param_.amaxList + 36;
      ProjBiasNorm_amax_ptr = param_.amaxList + 44;
      F1Bias_amax_ptr = param_.amaxList + 52;
      F2BiasNorm_amax_ptr = param_.amaxList + 60;
      Proj_aftergemm_amax_ptr = param_.amaxList + 40;
      F1_aftergemm_amax_ptr = param_.amaxList + 48;
      F2_aftergemm_amax_ptr = param_.amaxList + 56;
      to_tensor_amax_ptr = param_.amaxList;

      FC0_weight_amax_list =
          param_.amaxList + ACTIVATION_AMAX_NUM + 3 * hidden_dim;
      FC1_weight_amax_list = FC0_weight_amax_list + hidden_dim;
      FC2_weight_amax_list = FC1_weight_amax_list + 4 * hidden_dim;

      // This D2H copy operation will cause performance degradation
      if ((int8_mode_ == 1 && ((batch_size_ * from_seq_len_ >= 512) ||
                               (from_seq_len_ % 32 != 0))) ||
          int8_mode_ == 2 || int8_mode_ == 3) {
        // copy (int8O_gemm_deQ_scale_list + trt_fused_mha_amax_list) amax into
        // scale_list
        check_cuda_error(cudaMemcpyAsync(
            scale_list,
            FC2_weight_amax_list + hidden_dim,
            (INT8O_GEMM_NUM + TRT_FUSED_MHA_AMAX_NUM) * sizeof(float),
            cudaMemcpyDeviceToHost,
            param_.stream));
        int8O_gemm_deQ_scale_list = scale_list;
      }
      int k = hidden_dim;

      const int m = param_.sequence_id_offset == nullptr
                        ? batch_size_ * from_seq_len_
                        : param_.valid_word_num;
      if (layer_idx_ == 0) {
        genTransATensorAndInt8TensorForFirstLayer();
      } else {
        transA_from_tensor_ = param_.from_tensor;
        if (int8_mode_ == 2 || int8_mode_ == 3) {
          int8_from_tensor_ = (const int8_t *)transA_from_tensor_;
        } else if (int8_mode_ == 1) {
          quantized_kernelLauncher(int8_from_tensor_tmp_,
                                   transA_from_tensor_,
                                   m * k,
                                   to_tensor_amax_ptr + 3,
                                   param_.stream);
          int8_from_tensor_ = (const int8_t *)(int8_from_tensor_tmp_);
        }
      }

      multi_head_init_param.int8_from_tensor = int8_from_tensor_;

      multi_head_init_param.amaxList = param_.amaxList;

      multi_head_init_param.int8O_gemm_deQ_scale_list =
          int8O_gemm_deQ_scale_list;

      multi_head_init_param.trt_fused_mha_amax_list =
          scale_list + INT8O_GEMM_NUM;
    }

    multi_head_init_param.from_tensor = param.from_tensor;
    multi_head_init_param.to_tensor = param.to_tensor;
    multi_head_init_param.self_attention = param.self_attention;
    multi_head_init_param.attr_mask = param.attr_mask;
    multi_head_init_param.stream = param.stream;
    multi_head_init_param.cublas_handle = param.cublas_handle;
    multi_head_init_param.cublaslt_handle = param_.cublaslt_handle;
    multi_head_init_param.attr_out = attr_out_buf_;
    multi_head_init_param.valid_word_num = param.valid_word_num;
    multi_head_init_param.sequence_id_offset = param.sequence_id_offset;
    multi_head_init_param.trt_seqlen_offset = param_.trt_seqlen_offset;
    multi_head_init_param.trt_seqlen_size = param_.trt_seqlen_size;

    attention_->initialize(multi_head_init_param);
  }

  /**
   * do forward
   **/
  void forward() {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    try {
      attention_->forward();

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      DataType_ alpha = (DataType_)1.0f;
      DataType_ beta = (DataType_)0.0f;
      const int m = param_.sequence_id_offset == nullptr
                        ? batch_size_ * from_seq_len_
                        : param_.valid_word_num;
      int k = head_num_ * size_per_head_;
      int n = k;

      if (int8_mode_ != 0) {
        if (int8_mode_ == 1) {
          cublasLtMM_withAlgo(
              int_buf_,
              1,
              m,
              n,
              k,
              m * k,
              n * k,
              m * n,
              (int8_t *)attr_out_buf_,
              (int8_t *)(param_.self_attention.attention_output_weight.kernel),
              param_.cublaslt_handle,
              param_.stream,
              cublasAlgoMap_,
              use_ORDER_COL32_2R_4R4_);
          add_bias_input_layernorm_COL32_int32I_DataTypeO_kernelLauncher(
              attr_matmul_buf_,
              int_buf_,
              transA_from_tensor_,
              param_.self_attention.attention_output_weight.bias,
              param_.self_layernorm.gamma,
              param_.self_layernorm.beta,
              m,
              n,
              param_.stream,
              FC0_weight_amax_list,
              bmm2_amax_ptr);
        } else if (int8_mode_ == 2 || int8_mode_ == 3) {
          cublasLtMM_withAlgo_int8IO(
              (int8_t *)int_buf_,
              1,
              m,
              n,
              k,
              m * k,
              n * k,
              m * n,
              int8O_gemm_deQ_scale_list[5],
              (int8_t *)attr_out_buf_,
              (int8_t *)(param_.self_attention.attention_output_weight.kernel),
              param_.cublaslt_handle,
              param_.stream,
              cublasAlgoMap_,
              use_ORDER_COL32_2R_4R4_);
          add_bias_input_layernorm_COL32_int8IO_kernelLauncher(
              (int8_t *)attr_matmul_buf_,
              (int8_t *)int_buf_,
              int8_from_tensor_,
              param_.self_attention.attention_output_weight.bias,
              param_.self_layernorm.gamma,
              param_.self_layernorm.beta,
              m,
              n,
              param_.stream,
              Proj_aftergemm_amax_ptr + 1,
              to_tensor_amax_ptr + 1,
              ProjBiasNorm_amax_ptr + 3);
        }

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        n *= 4;

        if (int8_mode_ == 1) {
          quantized_kernelLauncher(attr_matmul_buf_tmp_,
                                   attr_matmul_buf_,
                                   k * m,
                                   ProjBiasNorm_amax_ptr + 3,
                                   param_.stream);
          cublasLtMM_withAlgo(int_buf_,
                              1,
                              m,
                              n,
                              k,
                              m * k,
                              n * k,
                              m * n,
                              attr_matmul_buf_tmp_,
                              (int8_t *)(param_.ffn.intermediate_weight.kernel),
                              param_.cublaslt_handle,
                              param_.stream,
                              cublasAlgoMap_,
                              use_ORDER_COL32_2R_4R4_);
          add_bias_act_COL32_int32I_int8O_kernelLauncher(
              (int8_t *)inter_matmul_buf_,
              int_buf_,
              param_.ffn.intermediate_weight.bias,
              m,
              n,
              param_.stream,
              FC1_weight_amax_list,
              ProjBiasNorm_amax_ptr + 2,
              F1Bias_amax_ptr + 3);
        } else if (int8_mode_ == 2 || int8_mode_ == 3) {
          cublasLtMM_withAlgo_int8IO(
              (int8_t *)int_buf_,
              1,
              m,
              n,
              k,
              m * k,
              n * k,
              m * n,
              int8O_gemm_deQ_scale_list[6],
              (int8_t *)attr_matmul_buf_,
              (int8_t *)(param_.ffn.intermediate_weight.kernel),
              param_.cublaslt_handle,
              param_.stream,
              cublasAlgoMap_,
              use_ORDER_COL32_2R_4R4_);
          add_bias_act_COL32_int8IO_kernelLauncher(
              (int8_t *)inter_matmul_buf_,
              (int8_t *)int_buf_,
              param_.ffn.intermediate_weight.bias,
              m,
              n,
              param_.stream,
              F1_aftergemm_amax_ptr + 1,
              F1Bias_amax_ptr + 3);
        }

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        n = k;
        k *= 4;

        if (int8_mode_ == 1) {
          cublasLtMM_withAlgo(int_buf_,
                              1,
                              m,
                              n,
                              k,
                              m * k,
                              n * k,
                              m * n,
                              (int8_t *)inter_matmul_buf_,
                              (int8_t *)(param_.ffn.output_weight.kernel),
                              param_.cublaslt_handle,
                              param_.stream,
                              cublasAlgoMap_,
                              use_ORDER_COL32_2R_4R4_);
          if (layer_idx_ != layer_num_ - 1) {
            add_bias_input_layernorm_COL32_int32I_DataTypeO_kernelLauncher(
                param_.transformer_out,
                int_buf_,
                attr_matmul_buf_,
                param_.ffn.output_weight.bias,
                param_.ffn_layernorm.gamma,
                param_.ffn_layernorm.beta,
                m,
                n,
                param_.stream,
                FC2_weight_amax_list,
                F1Bias_amax_ptr);
          } else {
            add_bias_input_layernorm_COL32_int32I_DataTypeO_kernelLauncher(
                transformer_out_tmp_DataType_,
                int_buf_,
                attr_matmul_buf_,
                param_.ffn.output_weight.bias,
                param_.ffn_layernorm.gamma,
                param_.ffn_layernorm.beta,
                m,
                n,
                param_.stream,
                FC2_weight_amax_list,
                F1Bias_amax_ptr);
            transposeMatrix_COL32ToColMajor_kernelLauncher(
                param_.transformer_out,
                transformer_out_tmp_DataType_,
                m,
                n,
                param_.stream);
          }
        } else if (int8_mode_ == 2 || int8_mode_ == 3) {
          cublasLtMM_withAlgo_int8IO(
              (int8_t *)int_buf_,
              1,
              m,
              n,
              k,
              m * k,
              n * k,
              m * n,
              int8O_gemm_deQ_scale_list[7],
              (int8_t *)inter_matmul_buf_,
              (int8_t *)(param_.ffn.output_weight.kernel),
              param_.cublaslt_handle,
              param_.stream,
              cublasAlgoMap_,
              use_ORDER_COL32_2R_4R4_);
          if (layer_idx_ != layer_num_ - 1) {
            add_bias_input_layernorm_COL32_int8IO_kernelLauncher(
                (int8_t *)param_.transformer_out,
                (int8_t *)int_buf_,
                (int8_t *)attr_matmul_buf_,
                param_.ffn.output_weight.bias,
                param_.ffn_layernorm.gamma,
                param_.ffn_layernorm.beta,
                m,
                n,
                param_.stream,
                F2_aftergemm_amax_ptr + 1,
                ProjBiasNorm_amax_ptr + 1,
                F2BiasNorm_amax_ptr + 3);
          } else {
            add_bias_input_layernorm_COL32_int8I_DataTypeO_kernelLauncher(
                transformer_out_tmp_DataType_,
                (int8_t *)int_buf_,
                (int8_t *)attr_matmul_buf_,
                param_.ffn.output_weight.bias,
                param_.ffn_layernorm.gamma,
                param_.ffn_layernorm.beta,
                m,
                n,
                param_.stream,
                F2_aftergemm_amax_ptr + 1,
                ProjBiasNorm_amax_ptr + 1);
            transposeMatrix_COL32ToColMajor_kernelLauncher(
                param_.transformer_out,
                transformer_out_tmp_DataType_,
                m,
                n,
                param_.stream);
          }
        }

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
      } else {
        cublasMM_cublasLtMM_wrapper(
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
            attr_out_buf_,
            BType_,
            k,
            &beta,
            (DataType_ *)attr_matmul_buf_,
            CType_,
            n,
            param_.stream,
            cublasAlgoMap_,
            sm_,
            cublas_workspace_);

        add_bias_input_layernorm_kernelLauncher<DataType_>(
            attr_matmul_buf_,
            param_.from_tensor,
            param_.self_attention.attention_output_weight.bias,
            param_.self_layernorm.gamma,
            param_.self_layernorm.beta,
            m,
            n,
            param_.stream);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        n *= 4;

        cublasMM_cublasLtMM_wrapper(param_.cublaslt_handle,
                                    param_.cublas_handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    n,
                                    m,
                                    k,
                                    &alpha,
                                    param_.ffn.intermediate_weight.kernel,
                                    AType_,
                                    n,
                                    attr_matmul_buf_,
                                    BType_,
                                    k,
                                    &beta,
                                    (DataType_ *)inter_matmul_buf_,
                                    CType_,
                                    n,
                                    param_.stream,
                                    cublasAlgoMap_,
                                    sm_,
                                    cublas_workspace_);
        if (use_gelu_ == true) {
          add_bias_act_kernelLauncher<DataType_>(
              inter_matmul_buf_,
              param_.ffn.intermediate_weight.bias,
              m,
              n,
              ActivationType::GELU,
              param_.stream);
        } else {
          add_bias_act_kernelLauncher<DataType_>(
              inter_matmul_buf_,
              param_.ffn.intermediate_weight.bias,
              m,
              n,
              ActivationType::RELU,
              param_.stream);
        }

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        n = k;
        k *= 4;

        cublasMM_cublasLtMM_wrapper(param_.cublaslt_handle,
                                    param_.cublas_handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    n,
                                    m,
                                    k,
                                    &alpha,
                                    param_.ffn.output_weight.kernel,
                                    AType_,
                                    n,
                                    inter_matmul_buf_,
                                    BType_,
                                    k,
                                    &beta,
                                    (DataType_ *)(param_.transformer_out),
                                    CType_,
                                    n,
                                    param_.stream,
                                    cublasAlgoMap_,
                                    sm_,
                                    cublas_workspace_);

        add_bias_input_layernorm_kernelLauncher<DataType_>(
            param_.transformer_out,
            attr_matmul_buf_,
            param_.ffn.output_weight.bias,
            param_.ffn_layernorm.gamma,
            param_.ffn_layernorm.beta,
            m,
            n,
            param_.stream);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
      }
    } catch (std::runtime_error &error) {
      throw error;
    }
  }

  ~BertEncoderTransformer() {
    if (buf_ != NULL) {
      if (allocator_ == NULL) {
        printf(
            "[ERROR][BertEncoderTransformer][~BertEncoderTransformer] "
            "allocator_ is NULL!\n");
        exit(-1);
      }
      allocator_->free(buf_);
    }
    if (attention_ != NULL) delete attention_;
  }
};

}  // namespace fastertransformer
