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

namespace fastertransformer {

template <typename T>
__global__ void transpose_cache_batch_major(T* k_dst,
                                            T* v_dst,
                                            const float* k_src,
                                            const float* v_src,
                                            const int* memory_seq_len,
                                            const int head_num,
                                            const int size_per_head,
                                            const int memory_max_seq_len,
                                            const int cache_max_len) {
  const int hidden_dim = head_num * size_per_head;
  const int x = (sizeof(T) == 4) ? 4 : 8;
  const int size_per_head_split = size_per_head / x;
  const int batch_id = blockIdx.x;
  const int seq_id = blockIdx.y;

  for (int id = threadIdx.x; id < head_num * size_per_head_split * x;
       id += blockDim.x) {
    int tmp_id = id;
    int x_id = tmp_id % x;
    tmp_id = (tmp_id - x_id) / x;
    int size_id = tmp_id % size_per_head_split;
    tmp_id = (tmp_id - size_id) / size_per_head_split;
    int head_id = tmp_id % head_num;

    int src_seq_id =
        (seq_id < memory_seq_len[batch_id])
            ? (seq_id + memory_max_seq_len - memory_seq_len[batch_id])
            : (seq_id - memory_seq_len[batch_id]);

    // key: [B, head_num, L, size_per_head / x, x] -> [B, head_num,
    // size_per_head / x, L, x]
    k_dst[batch_id * hidden_dim * cache_max_len +
          head_id * size_per_head * cache_max_len +
          size_id * cache_max_len * x + seq_id * x + x_id] =
        (T)k_src[batch_id * hidden_dim * memory_max_seq_len +
                 head_id * size_per_head * memory_max_seq_len +
                 src_seq_id * size_per_head + size_id * x + x_id];

    // value: [B, head_num, L, size_per_head/x, x] -> [B, head_num, L,
    // size_per_head/x, x]
    v_dst[batch_id * hidden_dim * cache_max_len +
          head_id * size_per_head * cache_max_len + seq_id * size_per_head +
          size_id * x + x_id] =
        (T)v_src[batch_id * hidden_dim * memory_max_seq_len +
                 head_id * size_per_head * memory_max_seq_len +
                 src_seq_id * size_per_head + size_id * x + x_id];
  }
}

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
                                      const int memory_max_seq_len,
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
  float local_i = (tid >= (memory_max_seq_len - memory_sequence_length[bid]) &&
                   (tid < step))
                      ? (float)logits[tid]
                      : -1e20f;
  float max_val = blockReduceMax<float>(local_i);
  if (tid == 0) s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = (tid >= (memory_max_seq_len - memory_sequence_length[bid]) &&
                   (tid < step))
                      ? __expf(local_i)
                      : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if (tid == 0) s_sum = val;  // + 1e-6;
  __syncthreads();

  if (tid >= (memory_max_seq_len - memory_sequence_length[bid]) &&
      (tid < step)) {
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
                             const int memory_max_seq_len,
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
          memory_max_seq_len,
          scalar);
#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
  }
}

template void self_attention_dispatch(const int* memory_sequence_length,
                                      float* key_buf,
                                      float* value_buf,
                                      float* query_buf,
                                      const float* self_Q_bias,
                                      float* key_cache,
                                      const float* self_K_bias,
                                      float* value_cache,
                                      const float* self_V_bias,
                                      float* context_buf,
                                      int batch_size,
                                      int head_num,
                                      int size_per_head,
                                      const int step,
                                      const int memory_max_seq_len,
                                      cudaStream_t stream);

template void self_attention_dispatch(const int* memory_sequence_length,
                                      half* key_buf,
                                      half* value_buf,
                                      half* query_buf,
                                      const half* self_Q_bias,
                                      half* key_cache,
                                      const half* self_K_bias,
                                      half* value_cache,
                                      const half* self_V_bias,
                                      half* context_buf,
                                      int batch_size,
                                      int head_num,
                                      int size_per_head,
                                      const int step,
                                      const int memory_max_seq_len,
                                      cudaStream_t stream);

template <typename T>
void transpose_cache_batch_major_kernelLauncher(T* k_dst,
                                                T* v_dst,
                                                const float* k_src,
                                                const float* v_src,
                                                const int* memory_seq_len,
                                                const int local_batch_size,
                                                const int memory_max_seq_len,
                                                const int cache_max_len,
                                                const int size_per_head,
                                                const int local_head_num,
                                                cudaStream_t stream) {
  constexpr int block_sz = 128;
  dim3 grid(local_batch_size, memory_max_seq_len);

  transpose_cache_batch_major<<<grid, block_sz, 0, stream>>>(k_dst,
                                                             v_dst,
                                                             k_src,
                                                             v_src,
                                                             memory_seq_len,
                                                             local_head_num,
                                                             size_per_head,
                                                             memory_max_seq_len,
                                                             cache_max_len);
}

template void transpose_cache_batch_major_kernelLauncher(
    float* k_dst,
    float* v_dst,
    const float* k_src,
    const float* v_src,
    const int* memory_seq_len,
    const int local_batch_size,
    const int memory_max_seq_len,
    const int cache_max_len,
    const int size_per_head,
    const int local_head_num,
    cudaStream_t stream);

template void transpose_cache_batch_major_kernelLauncher(
    half* k_dst,
    half* v_dst,
    const float* k_src,
    const float* v_src,
    const int* memory_seq_len,
    const int local_batch_size,
    const int memory_max_seq_len,
    const int cache_max_len,
    const int size_per_head,
    const int local_head_num,
    cudaStream_t stream);
}
