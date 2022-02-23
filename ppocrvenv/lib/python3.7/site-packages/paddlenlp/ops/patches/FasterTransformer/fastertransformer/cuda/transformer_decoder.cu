/*
 * Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

namespace fastertransformer {

template <typename T>
__global__ void self_attention_kernel(const int* memory_sequence_length,
                                      T* key_buf,
                                      T* value_buf,
                                      T* query_buf,
                                      const T* self_Q_bias,
                                      T* key_cache,
                                      const T* self_K_bias,
                                      T* value_cache,
                                      const T* self_V_bias,
                                      T* context_buf,
                                      int batch_size,
                                      int head_num,
                                      int size_per_head,
                                      const int step,
                                      const int start_len,
                                      const T scalar) {
  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T*>(s_buf);
  T* logits = reinterpret_cast<T*>(&sq[size_per_head]);

  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  int qkv_bias_id = head_id * size_per_head + tid;

  if (tid < size_per_head)
    sq[tid] = query_buf[qkv_id] + self_Q_bias[qkv_bias_id];
  __syncthreads();

  // offset for each step
  int offset = batch_size * head_num * size_per_head;
  for (int ite = 0; ite < step; ++ite) {
    T key = tid < size_per_head ? key_cache[ite * offset + qkv_id] : (T)0.0f;
    // for the last step, we should update K + bias_K to the cache
    if (ite == step - 1 && tid < size_per_head) {
      key = key_buf[qkv_id] + self_K_bias[qkv_bias_id];
      key_cache[ite * offset + qkv_id] = key;
    }

    T val = (tid < size_per_head) ? key * sq[tid] * (T)(scalar) : (T)(0.0f);
    T qk = blockReduceSum(val);
    if (threadIdx.x == 0) {
      logits[ite] = qk;
    }
    __syncthreads();  // try to remove
  }
  __syncthreads();  // try to remove

  __shared__ float s_max_val, s_sum;
  float local_i =
      (tid >= (start_len - memory_sequence_length[bid]) && (tid < step))
          ? (float)logits[tid]
          : -1e20f;
  float max_val = blockReduceMax<float>(local_i);
  if (tid == 0) s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o =
      (tid >= (start_len - memory_sequence_length[bid]) && (tid < step))
          ? __expf(local_i)
          : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if (tid == 0) s_sum = val;  // + 1e-6;
  __syncthreads();

  if (tid >= (start_len - memory_sequence_length[bid]) && (tid < step)) {
    logits[tid] = local_o / s_sum;
  } else if (tid < step) {
    logits[tid] = static_cast<T>(0.0f);
  }
  __syncthreads();

  if (tid < size_per_head) {
    T sum = (T)0.0f;
    for (int ite = 0; ite < step; ++ite) {
      T value = value_cache[ite * offset + qkv_id];
      // for the last step, we should update V + bias_V to the cache
      if (ite == step - 1) {
        value = value_buf[qkv_id] + self_V_bias[qkv_bias_id];
        value_cache[ite * offset + qkv_id] = value;
      }
      sum += value * logits[ite];
    }
    context_buf[qkv_id] = sum;
  }
}


template <typename T>
void self_attention_dispatch(const int* memory_sequence_length,
                             T* key_buf,
                             T* value_buf,
                             T* query_buf,
                             const T* self_Q_bias,
                             T* key_cache,
                             const T* self_K_bias,
                             T* value_cache,
                             const T* self_V_bias,
                             T* context_buf,
                             int batch_size,
                             int head_num,
                             int size_per_head,
                             const int step,
                             const int start_len,
                             cudaStream_t stream) {
  const int block_sz = ATTENTION_BLOCK_SIZE;
  T scalar = (T)(1.f / sqrtf(size_per_head * 1.0f));

  dim3 grid(batch_size * head_num);

  int cond = size_per_head * ((ATTENION_OPT) ? 1 : 0);
  switch (cond) {
    /*case 32:
      masked_attention_kernel_opt<32, block_sz, T><<<grid, block_sz,
    sizeof(float)*step, stream>>>(
        key_buf, value_buf,
        query_buf, self_Q_bias,  key_cache, self_K_bias, value_cache,
    self_V_bias, context_buf,
        batch_size, head_num, step, scalar);
      break;
    case 64:
      if(sizeof(T) == 2)
        masked_attention_kernel_opt_half2<64, block_sz><<<grid, block_sz,
    sizeof(float)*step, stream>>>(
          key_buf, value_buf,
          query_buf, self_Q_bias,  key_cache, self_K_bias, value_cache,
    self_V_bias, context_buf,
          batch_size, head_num, step, scalar);
      else
        masked_attention_kernel_opt<64, block_sz, T><<<grid, block_sz,
    sizeof(float)*step, stream>>>(
          key_buf, value_buf,
          query_buf, self_Q_bias,
          key_cache, self_K_bias,
          value_cache, self_V_bias,
          context_buf,
          batch_size, head_num, step, scalar);
      break;
    case 128:
      if(sizeof(T) == 2)
        masked_attention_kernel_opt_half2<128, block_sz><<<grid, block_sz,
    sizeof(float)*step, stream>>>(
          key_buf, value_buf,
          query_buf, self_Q_bias,  key_cache, self_K_bias, value_cache,
    self_V_bias, context_buf,
          batch_size, head_num, step, scalar);
      else
        masked_attention_kernel_opt<128, block_sz, T><<<grid, block_sz,
    sizeof(float)*step, stream>>>(
          key_buf, value_buf,
          query_buf, self_Q_bias,  key_cache, self_K_bias, value_cache,
    self_V_bias, context_buf,
          batch_size, head_num, step, scalar);
      break;*/
    default:
      // default path
      int block_size = 128;

      // suppose size_per_head <= 128
      if (step <= 64)
        block_size = 64;
      else if (step <= 128 && step > size_per_head)
        block_size = 128;
      else if (step > 128 && step <= 256)
        block_size = 256;
      else if (step > 256 && step <= 512)
        block_size = 512;
      else
        block_size = 1024;

      if ((int)block_size < size_per_head) {
        block_size = size_per_head;
      }

      assert(block_size <= 1024);
      dim3 block(block_size);
      T scalar = 1 / sqrtf(size_per_head * 1.0f);

      int shared_size = sizeof(T) * (size_per_head + step);
      self_attention_kernel<T><<<grid, block, shared_size, stream>>>(
          memory_sequence_length,
          key_buf,
          value_buf,
          query_buf,
          self_Q_bias,
          key_cache,
          self_K_bias,
          value_cache,
          self_V_bias,
          context_buf,
          batch_size,
          head_num,
          size_per_head,
          step,
          start_len,
          scalar);
#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
  }
}


template <OperationType OpType_>
void OpenTransformerDecoder<OpType_>::self_multi_head_attention(
    const DataType_* from_tensor,
    const int* memory_sequence_length,
    DataType_* key_cache_,
    DataType_* value_cache_,
    DataType_* decoder_output,
    const int step,
    const int start_len) {
  int m = batch_size_;
  int n = hidden_units_;
  int k = hidden_units_;

  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

  if (is_fuse_QKV == true) {
    check_cuda_error(
        cublasGemmBatchedEx(param_.cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            (const void* const*)qkv_kernel_,
                            AType_,
                            n,
                            (const void* const*)qkv_input_,
                            BType_,
                            k,
                            &beta,
                            (void* const*)qkv_buf_,
                            CType_,
                            n,
                            3,
                            computeType_,
                            static_cast<cublasGemmAlgo_t>(cublasAlgo_[4])));
  } else {
    key_buf_ = key_cache_ + (step - 1) * m * n;
    value_buf_ = value_cache_ + (step - 1) * m * n;

    check_cuda_error(
        cublasGemmEx(param_.cublas_handle,
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
                     computeType_,
                     static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

    check_cuda_error(
        cublasGemmEx(param_.cublas_handle,
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
                     computeType_,
                     static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

    check_cuda_error(
        cublasGemmEx(param_.cublas_handle,
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
                     computeType_,
                     static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
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
                                     batch_size_,
                                     head_num_,
                                     size_per_head_,
                                     step,
                                     start_len,
                                     param_.stream);

  check_cuda_error(
      cublasGemmEx(param_.cublas_handle,
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
                   computeType_,
                   static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
}


template <OperationType OpType_>
void OpenTransformerDecoder<OpType_>::decoder_norm1(const DataType_* input,
                                                    const DataType_* gamma,
                                                    const DataType_* beta,
                                                    DataType_* output,
                                                    int m,
                                                    int n) {
  dim3 grid(m);
  dim3 block(min(n, 1024));

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
     Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */
  if (n % 32 != 0) block.x = 1024;

  block.x =
      block.x /
      (4 / sizeof(DataType_));  // if using half, only need half of block.x

  /* should pay attention to the rsqrt precision*/
  // assert(block.x <= 1024);
  // decoder_norm1_kernel<DataType_><<<grid, block, 0, param_.stream>>>(input,
  // gamma, beta, output, m, n);
  decoder_norm1_kernel_generalize<DataType_><<<grid, block, 0, param_.stream>>>(
      input, gamma, beta, output, m, n);  // For gpt-3
}

template <OperationType OpType_>
void OpenTransformerDecoder<OpType_>::decoder_norm2(const DataType_* input,
                                                    const DataType_* gamma,
                                                    const DataType_* beta,
                                                    const DataType_* bias,
                                                    DataType_* output,
                                                    DataType_* norm_output,
                                                    int m,
                                                    int n) {
  dim3 grid(m);
  dim3 block(min(n, 1024));


  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
  Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */

  if (n % 32 != 0) block.x = 1024;

  block.x =
      block.x /
      (4 / sizeof(DataType_));  // if using half, only need half of block.x

  /* should pay attention to the rsqrt precision*/
  // assert(block.x <= 1024);
  // decoder_norm2_kernel<DataType_><<<grid, block, 0, param_.stream>>>(input,
  // gamma, beta, bias, output, norm_output, m, n);
  decoder_norm2_kernel_generalize<DataType_><<<grid, block, 0, param_.stream>>>(
      input, gamma, beta, bias, output, norm_output, m, n);  // For gpt-3
}

template <OperationType OpType_>
void OpenTransformerDecoder<OpType_>::ffn(const DataType_* input,
                                          DataType_* ffn_inner,
                                          DataType_* output,
                                          const int m,
                                          const int inner_size,
                                          const int n,
                                          ActivationType activation_type) {
  int m1 = m, k1 = n, n1 = inner_size;
  DataType_ alpha = (DataType_)1.0f;
  DataType_ beta = (DataType_)0.0f;

  check_cuda_error(cublasGemmEx(param_.cublas_handle,
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
                                computeType_,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo_[2])));

  // dim3 grid(min(m1, 65536));
  // dim3 block(min(n1 / 4, 1024));

  // // TODO remove this limitation
  // // assert(block.x <= 1024);

  // if(activation_type == ActivationType::RELU)
  //   add_bias_relu<DataType_><<<grid, block, 0, param_.stream>>>(ffn_inner,
  //   param_.ffn.intermediate_weight.bias, m1, n1);
  // else if(activation_type == ActivationType::GELU)
  //   add_bias_gelu<DataType_><<<grid, block, 0, param_.stream>>>(ffn_inner,
  //   param_.ffn.intermediate_weight.bias, m1, n1);

  dim3 block(min((int)(n1 / 4 / (4 / sizeof(DataType_))), 1024));
  dim3 grid(min(m1 * n1 / block.x, 65536));

  if (activation_type == ActivationType::RELU)
    add_bias_relu<DataType_><<<grid, block, 0, param_.stream>>>(
        ffn_inner,
        param_.ffn.intermediate_weight.bias,
        m1,
        n1 / (4 / sizeof(DataType_)));
  else if (activation_type == ActivationType::GELU)
    add_bias_gelu<DataType_><<<grid, block, 0, param_.stream>>>(
        ffn_inner,
        param_.ffn.intermediate_weight.bias,
        m1,
        n1 / (4 / sizeof(DataType_)));


  int m2 = m, n2 = n, k2 = inner_size;
  check_cuda_error(cublasGemmEx(param_.cublas_handle,
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
                                computeType_,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo_[3])));
}

template <OperationType OpType_>
void OpenTransformerDecoder<OpType_>::add_bias_act(
    DataType_* input,
    const DataType_* bias,
    int m,
    int n,
    cudaStream_t stream,
    ActivationType activation_type = ActivationType::GELU) {
  dim3 block_(min((int)(n / 4 / (4 / sizeof(DataType_))), 1024));
  dim3 grid_(min(m * n / block_.x, 65536));

  if (activation_type == ActivationType::RELU)
    add_bias_relu<DataType_><<<grid_, block_, 0, stream>>>(
        input, bias, m, n / (4 / sizeof(DataType_)));
  else if (activation_type == ActivationType::GELU)
    add_bias_gelu<DataType_><<<grid_, block_, 0, stream>>>(
        input, bias, m, n / (4 / sizeof(DataType_)));
}

template <OperationType OpType_>
void OpenTransformerDecoder<OpType_>::add_bias_input(DataType_* output,
                                                     const DataType_* input,
                                                     const int m,
                                                     const int n) {
  dim3 grid(min(m, 65536));
  dim3 block(min(n, 1024));

  add_bias_input_kernel_generalize<<<grid, block, 0, param_.stream>>>(
      output, input, param_.ffn.output_weight.bias, m, n);
}

template void
OpenTransformerDecoder<OperationType::FP32>::self_multi_head_attention(
    const float* from_tensor,
    const int* memory_sequence_length,
    float* key_cache,
    float* value_cache,
    float* decoder_output,
    const int step,
    const int start_len);

template void
OpenTransformerDecoder<OperationType::FP16>::self_multi_head_attention(
    const half* from_tensor,
    const int* memory_sequence_length,
    half* key_cache,
    half* value_cache,
    half* decoder_output,
    const int step,
    const int start_len);

template void OpenTransformerDecoder<OperationType::FP32>::ffn(
    const float* input,
    float* ffn_inner,
    float* otuput,
    const int m,
    const int inner_size,
    const int n,
    ActivationType activation_type);

template void OpenTransformerDecoder<OperationType::FP16>::ffn(
    const half* input,
    half* ffn_inner,
    half* otuput,
    const int m,
    const int inner_size,
    const int n,
    ActivationType activation_type);

template void OpenTransformerDecoder<OperationType::FP32>::decoder_norm1(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int m,
    int n);

template void OpenTransformerDecoder<OperationType::FP16>::decoder_norm1(
    const half* input,
    const half* gamma,
    const half* beta,
    half* output,
    int m,
    int n);

template void OpenTransformerDecoder<OperationType::FP32>::decoder_norm2(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* bias,
    float* output,
    float* norm_output,
    int m,
    int n);

template void OpenTransformerDecoder<OperationType::FP16>::decoder_norm2(
    const half* input,
    const half* gamma,
    const half* beta,
    const half* bias,
    half* output,
    half* norm_output,
    int m,
    int n);

template void OpenTransformerDecoder<OperationType::FP32>::add_bias_act(
    float* input,
    const float* bias,
    int m,
    int n,
    cudaStream_t stream,
    ActivationType activation_type);

template void OpenTransformerDecoder<OperationType::FP16>::add_bias_act(
    half* input,
    const half* bias,
    int m,
    int n,
    cudaStream_t stream,
    ActivationType activation_type);

template void OpenTransformerDecoder<OperationType::FP32>::add_bias_input(
    float* output, const float* input, const int m, const int n);

template void OpenTransformerDecoder<OperationType::FP16>::add_bias_input(
    half* output, const half* input, const int m, const int n);


}  // namespace FasterTransformer
