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

#include <random>
#include "cub/cub.cuh"
#include "fastertransformer/cuda/topk_kernels.cuh"

namespace fastertransformer {

__global__ void ker_curand_setup(curandState_t* state,
                                 const int size,
                                 const int seed) {
  // curand_init(clock(), blockIdx.x * blockDim.x + threadIdx.x, 0,
  // &state[blockIdx.x * blockDim.x + threadIdx.x]);
  // fix the seed to prevent the seed of different gpu are differnet in Tensor
  // Parallel
  if (threadIdx.x + blockIdx.x * blockDim.x < size)
    curand_init(seed,
                blockIdx.x * blockDim.x + threadIdx.x,
                seed,
                &state[blockIdx.x * blockDim.x + threadIdx.x]);
}

void ker_curand_setupLauncher(curandState_t* state,
                              DecodingSamplingArguments args,
                              cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((int)(ceil(args.batch_size_ * 1.0 / 256)));
  int seed = clock();
  ker_curand_setup<<<grid, block, 0, stream>>>(state, args.batch_size_, seed);
}


template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void beam_topK_kernel(const T* log_probs,
                          int* topk_tmp_id_buf,
                          T* topk_tmp_val_buf,
                          const int vocab_size,
                          T diversity_rate) {
  typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  TopK<T, MAX_K> partial;

  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

#pragma unroll
  for (int i = 0; i < MAX_K; ++i) {
    partial.p[i] = -1;
    partial.u[i] = -MAX_T_VAL;
  }

#pragma unroll
  for (int elem_id = thread_id; elem_id < vocab_size;
       elem_id += THREADBLOCK_SIZE) {
    int index = elem_id + block_id * vocab_size;
    partial.insert(log_probs[index], index);
  }

  TopK<T, MAX_K> total =
      BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

  if (thread_id == 0) {
    int index = block_id * MAX_K;

#pragma unroll
    for (int i = 0; i < MAX_K; ++i) {
      topk_tmp_id_buf[index + i] = total.p[i];
      topk_tmp_val_buf[index + i] = total.u[i] + diversity_rate * (T)i;
    }
  }
}

template <typename T, int K>
__forceinline__ __device__ T blockRoughTopK(T val);

template <typename T, int beam_size, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void beam_topK_kernel_hierarchical(const T* log_probs,
                                       T* can_score_buf,
                                       int* can_idx_buf,
                                       int* topk_tmp_id_buf,
                                       T* topk_tmp_val_buf,
                                       const int vocab_size,
                                       T diversity_rate) {
  __shared__ T s_topk;
  __shared__ int num_cur_beam_can;
  typedef cub::BlockReduce<TopK<T, beam_size>, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
  T rough_top_kth_logit = -MAX_T_VAL;

#pragma unroll
  for (int elem_id = thread_id; elem_id < vocab_size;
       elem_id += THREADBLOCK_SIZE) {
    int index = elem_id + block_id * vocab_size;
    rough_top_kth_logit = fmaxf(rough_top_kth_logit, log_probs[index]);
  }
  rough_top_kth_logit = blockRoughTopK<float, beam_size>(rough_top_kth_logit);
  if (thread_id == 0) {
    s_topk = rough_top_kth_logit;
    num_cur_beam_can = 0;
  }

  int idx = block_id * vocab_size + thread_id;

  __shared__ int l_n;  // current iteration candidate number
  for (int iter = 0;
       iter < (vocab_size + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE;
       iter++) {
    // zero the counter
    if (threadIdx.x == 0) l_n = 0;
    __syncthreads();
    T lgt = -MAX_T_VAL;  // min s_topk is CUDA_FLOAT_INF_NEG
    int pos;
    int vocab_id = idx - block_id * vocab_size;

    if (vocab_id < vocab_size) {
      lgt = log_probs[idx];
      if (lgt >= s_topk) pos = atomicAdd(&l_n, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      l_n = atomicAdd(&num_cur_beam_can, l_n);
    }
    __syncthreads();

    if (lgt >= s_topk) {
      pos += l_n;
      can_score_buf[pos + block_id * vocab_size] = lgt;
      can_idx_buf[pos + block_id * vocab_size] = idx;
    }
    __syncthreads();
    idx += THREADBLOCK_SIZE;
  }

  TopK<T, beam_size> partial;
#pragma unroll
  for (int i = 0; i < beam_size; ++i) {
    partial.p[i] = -1;
    partial.u[i] = -MAX_T_VAL;
  }
  for (int elem_id = thread_id; elem_id < num_cur_beam_can;
       elem_id += THREADBLOCK_SIZE) {
    int index = elem_id + block_id * vocab_size;
    partial.insert(can_score_buf[index], index);
  }
  TopK<T, beam_size> total =
      BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, beam_size>);

  if (thread_id == 0) {
    int index = block_id * beam_size;

#pragma unroll
    for (int i = 0; i < beam_size; ++i) {
      topk_tmp_id_buf[index + i] = can_idx_buf[total.p[i]];
      topk_tmp_val_buf[index + i] = total.u[i] + diversity_rate * (T)i;
    }
  }
}

template <typename T, int THREADBLOCK_SIZE>
__global__ void beam_topK_kernel_general(const T* log_probs,
                                         T* tmp_log_probs,
                                         int* topk_tmp_id_buf,
                                         T* topk_tmp_val_buf,
                                         const int k,
                                         const int vocab_size) {
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
  typedef cub::BlockReduce<TopK_2<T>, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  TopK_2<T> partial;

  for (int elem_id = tid; elem_id < vocab_size; elem_id += THREADBLOCK_SIZE) {
    int index = elem_id + bid * vocab_size;
    tmp_log_probs[index] = log_probs[index];
  }

  for (int ite = 0; ite < k; ite++) {
    partial.init();
#pragma unroll
    for (int elem_id = tid; elem_id < vocab_size; elem_id += THREADBLOCK_SIZE) {
      int index = elem_id + bid * vocab_size;
      partial.insert(tmp_log_probs[index], index);
    }

    TopK_2<T> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

    if (tid == 0) {
      const int index = bid * k + ite;
      topk_tmp_id_buf[index] = total.p;
      topk_tmp_val_buf[index] = total.u;
      tmp_log_probs[total.p] = -MAX_T_VAL;
    }
    __syncthreads();
  }
}

#define CASE_K(K)                                                              \
  case K:                                                                      \
    beam_topK_kernel<T, K, block_size><<<batch_size, block_size, 0, stream>>>( \
        log_probs, topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, 0.0f);       \
    break;

template <typename T>
void beam_topK_kernelLauncher(const T* log_probs,
                              int* topk_tmp_id_buf,
                              T* topk_tmp_val_buf,
                              DecodingSamplingArguments args,
                              cudaStream_t stream) {
  const int batch_size = args.batch_size_;
  const int vocab_size = args.vocab_size_padded_;
  const int candidate_num = args.candidate_num_;
  const int block_size = 256;
  switch (candidate_num) {
    CASE_K(1);
    CASE_K(2);
    CASE_K(4);
    default:
      printf("[ERROR] Topk kernel does not support candidate_num = %d \n",
             candidate_num);
      exit(0);
      break;
  }
}

#undef CASE_K

template void beam_topK_kernelLauncher(const float* log_probs,
                                       int* topk_tmp_id_buf,
                                       float* topk_tmp_val_buf,
                                       DecodingSamplingArguments args,
                                       cudaStream_t stream);

template void beam_topK_kernelLauncher(const half* log_probs,
                                       int* topk_tmp_id_buf,
                                       half* topk_tmp_val_buf,
                                       DecodingSamplingArguments args,
                                       cudaStream_t stream);

template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void batch_topK_kernel(int* topk_tmp_id_buf,
                           T* topk_tmp_val_buf,
                           int* id_buf) {
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
  TopK<T, MAX_K> partial;
  if (thread_id == 0) {
    for (int i = 0; i < MAX_K; ++i) {
      partial.p[i] = -1;
      partial.u[i] = -MAX_T_VAL;
    }

    int index = block_id * MAX_K * MAX_K;
    for (int i = 0; i < MAX_K * MAX_K; i++) {
      partial.insert((T)topk_tmp_val_buf[index + i],
                     topk_tmp_id_buf[index + i]);
    }

    index = block_id * MAX_K;
    for (int i = 0; i < MAX_K; i++) {
      id_buf[index + i] = partial.p[i];
    }
  }
}

template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void batch_topK_kernel_v2(int* topk_tmp_id_buf,
                              T* topk_tmp_val_buf,
                              int* id_buf) {
  typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  TopK<T, MAX_K> partial;
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

#pragma unroll
  for (int i = 0; i < MAX_K; ++i) {
    partial.p[i] = -1;
    partial.u[i] = -MAX_T_VAL;
  }

  int ite = MAX_K * MAX_K / THREADBLOCK_SIZE;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int index = bid * MAX_K * MAX_K + i * THREADBLOCK_SIZE + tid;
    partial.insert((T)topk_tmp_val_buf[index], topk_tmp_id_buf[index]);
  }

  TopK<T, MAX_K> total =
      BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < MAX_K; i++) id_buf[bid * MAX_K + i] = total.p[i];
  }
}

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_1_opt3(const T* __restrict log_probs,
                                  T* tmp_log_probs,
                                  int* topk_tmp_id_buf,
                                  T* topk_tmp_val_buf,
                                  const bool* finished,
                                  const int k,
                                  const int vocab_size,
                                  const int end_id) {
  typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  const int row_id = bid / BLOCKS_PER_BEAM_;      // row id for log_probs
  const int block_lane = bid % BLOCKS_PER_BEAM_;  // block id for a beam
  const int tmp_log_buf_index = row_id * vocab_size;
  const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM_ * k + block_lane * k;
  TopK_2<T> partial;
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

  if (finished != nullptr && finished[row_id] == true) {
    if (tid < k) {
      const int index = tmp_topk_buf_index + tid;
      if (block_lane == 0 && tid == 0) {
        topk_tmp_id_buf[index] = tmp_log_buf_index + end_id;
        topk_tmp_val_buf[index] = log_probs[tmp_log_buf_index + end_id];
      } else {
        topk_tmp_id_buf[index] = -1;
        topk_tmp_val_buf[index] = -MAX_T_VAL;
      }
    }
    return;
  }

  for (int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size;
       elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_) {
    int index = elem_id + tmp_log_buf_index;
    tmp_log_probs[index] = log_probs[index];
  }

  for (int ite = 0; ite < k; ite++) {
    partial.init();
#pragma unroll
    for (int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size;
         elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_) {
      int index = elem_id + tmp_log_buf_index;
      partial.insert(tmp_log_probs[index], index);
    }

    TopK_2<T> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

    if (tid == 0) {
      const int index = tmp_topk_buf_index + ite;
      topk_tmp_id_buf[index] = total.p;
      topk_tmp_val_buf[index] = total.u;
      tmp_log_probs[total.p] = -MAX_T_VAL;
    }
    __syncthreads();
  }
}

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_2_opt3(const int* __restrict topk_tmp_id_buf,
                                  T* topk_tmp_val_buf,
                                  int* ids,
                                  const int k) {
  const int size = k * k * BLOCKS_PER_BEAM_;
  const int tid = threadIdx.x;
  const int batch_id = blockIdx.x;
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

  typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  extern __shared__ char array[];
  T* s_val = topk_tmp_val_buf + batch_id * size;
  int* s_id = (int*)(array);

  TopK_2<T> partial;

  for (int ite = 0; ite < k; ite++) {
    partial.init();
#pragma unroll
    for (int i = tid; i < size; i += BLOCK_SIZE_) {
      partial.insert(s_val[i], i);
    }

    TopK_2<T> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

    if (tid == 0) {
      s_id[ite] = total.p;
      s_val[total.p] = -MAX_T_VAL;
    }
    __syncthreads();
  }
  if (tid < k)
    ids[batch_id * k + tid] = topk_tmp_id_buf[batch_id * size + s_id[tid]];
}

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_1_opt3(
    const T* __restrict log_probs,
    const float* __restrict cum_log_probs,  // If null, log_probs is
                                            // cum_log_probs.
    // Used in beam_search_v2 which adding
    // cum_log_buf_ to logits_buf_ here.
    T* tmp_log_probs,
    int* topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    const bool* finished,
    const int k,
    const int vocab_size,
    const T
        diversity_rate,  // diversity_rate only works when BLOCKS_PER_BEAM_ is 1
    const int end_id) {
  typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  const int row_id = bid / BLOCKS_PER_BEAM_;      // row id for log_probs
  const int block_lane = bid % BLOCKS_PER_BEAM_;  // block id for a beam
  const int tmp_log_buf_index = row_id * vocab_size;
  const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM_ * k + block_lane * k;
  const int beam_id_in_output =
      row_id / (k >> 1) * k + row_id % (k >> 1) + (k >> 1);
  TopK_2<T> partial;
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

  for (int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size;
       elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_) {
    int index = elem_id + tmp_log_buf_index;
    tmp_log_probs[index] = log_probs[index];
  }

  for (int ite = 0; ite < k; ite++) {
    partial.init();
#pragma unroll
    for (int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size;
         elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_) {
      int index = elem_id + tmp_log_buf_index;
      partial.insert(tmp_log_probs[index], index);
    }

    TopK_2<T> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

    if (tid == 0) {
      const int index = tmp_topk_buf_index + ite;
      topk_tmp_id_buf[index] = total.p;
      topk_tmp_val_buf[index] =
          (cum_log_probs
               ? (T)((float)total.u + cum_log_probs[beam_id_in_output])
               : total.u) +
          diversity_rate * (T)ite;
      tmp_log_probs[total.p] = -MAX_T_VAL;
    }
    __syncthreads();
  }
}

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_2_opt3_sampling(
    const int* __restrict topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    T* topk_tmp2_val_buf,
    int* ids,
    int* sequence_length,
    bool* finished_buf,
    const int k,
    curandState_t* curandstate,
    const int end_id,
    const int vocab_size) {
  const int size = k * BLOCKS_PER_BEAM_;
  const int tid = threadIdx.x;
  const int batch_id = blockIdx.x;
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

  typedef cub::BlockReduce<TopK_2<float>, BLOCK_SIZE_> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  extern __shared__ char array[];
  __shared__ float rand_num;
  __shared__ float s_sum;
  __shared__ float s_max;
  T* s_val = topk_tmp_val_buf + batch_id * size;
  int* s_id = (int*)(array);
  s_max = (float)0.0f;
  s_sum = (float)0.0f;
  TopK_2<float> partial;

  for (int index = tid; index < size; index += BLOCK_SIZE_) {
    topk_tmp2_val_buf[batch_id * size + index] =
        topk_tmp_val_buf[batch_id * size + index];
  }
  __syncthreads();
  T* s_val2 = topk_tmp2_val_buf + batch_id * size;

  for (int ite = 0; ite < k; ite++) {
    partial.init();
#pragma unroll
    for (int i = tid; i < size; i += BLOCK_SIZE_) {
      partial.insert((float)s_val[i], i);
    }

    TopK_2<float> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<float>);

    if (ite == 0) s_max = total.u;

    if (tid == 0) {
      s_id[ite] = total.p;
      s_val[total.p] = -MAX_T_VAL;
      total.u = __expf(total.u - s_max);
      s_val2[total.p] = (T)total.u;
      s_sum += total.u;
    }
    __syncthreads();
  }
  if (tid == 0) {
    rand_num = (float)curand_uniform(curandstate + blockIdx.x) * s_sum;
    for (int i = 0; i < k; i++) {
      rand_num = rand_num - (float)s_val2[s_id[i]];
      if (rand_num <= 0.0f || i == k - 1) {
        ids[batch_id] = topk_tmp_id_buf[batch_id * size + s_id[i]] % vocab_size;
        break;
      }
    }
    if (finished_buf != nullptr) {
      if (sequence_length != nullptr) {
        sequence_length[batch_id] = finished_buf[batch_id]
                                        ? sequence_length[batch_id]
                                        : sequence_length[batch_id] + 1;
      }
      finished_buf[batch_id] = ids[batch_id] == end_id ? 1 : 0;
    }
  }
}

template <typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
__global__ void topk_stage_1_opt2_general(const T* __restrict log_probs,
                                          T* tmp_log_probs,
                                          int* topk_tmp_id_buf,
                                          T* topk_tmp_val_buf,
                                          const int k,
                                          const int vocab_size) {
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
  typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int row_id = bid / BLOCKS_PER_BEAM;      // row id for log_probs
  const int block_lane = bid % BLOCKS_PER_BEAM;  // block id for a beam
  const int tmp_log_buf_index = row_id * vocab_size;
  const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM * k + block_lane * k;
  TopK_2<T> partial;

  for (int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size;
       elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM) {
    int index = elem_id + tmp_log_buf_index;
    tmp_log_probs[index] = log_probs[index];
  }


  for (int ite = 0; ite < k; ite++) {
    partial.init();
#pragma unroll
    for (int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size;
         elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM) {
      int index = elem_id + tmp_log_buf_index;
      partial.insert(tmp_log_probs[index], index);
    }

    TopK_2<T> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

    if (tid == 0) {
      const int index = tmp_topk_buf_index + ite;
      topk_tmp_id_buf[index] = total.p;
      topk_tmp_val_buf[index] = total.u;
      tmp_log_probs[total.p] = -MAX_T_VAL;
    }
    __syncthreads();
  }
}

template <typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
__global__ void topk_stage_2_opt2_general(const int* __restrict topk_tmp_id_buf,
                                          T* topk_tmp_val_buf,
                                          int* ids,
                                          const int k) {
  const int size = k * k * BLOCKS_PER_BEAM;
  const int tid = threadIdx.x;
  const int batch_id = blockIdx.x;
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

  typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  extern __shared__ char array[];
  T* s_val = topk_tmp_val_buf + batch_id * size;
  int* s_id = (int*)(array);

  TopK_2<T> partial;

  for (int ite = 0; ite < k; ite++) {
    partial.init();
#pragma unroll
    for (int i = tid; i < size; i += BLOCK_SIZE) {
      partial.insert(s_val[i], i);
    }

    TopK_2<T> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

    if (tid == 0) {
      s_id[ite] = total.p;
      s_val[total.p] = -MAX_T_VAL;
    }
    __syncthreads();
  }
  if (tid < k)
    ids[batch_id * k + tid] = topk_tmp_id_buf[batch_id * size + s_id[tid]];
}

#define CASE_K_DIV(K, BLOCK_SIZE_1, BLOCK_SIZE_2)                            \
  case K:                                                                    \
    beam_topK_kernel<                                                        \
        T,                                                                   \
        K,                                                                   \
        BLOCK_SIZE_2><<<batch_size * beam_width, BLOCK_SIZE_2, 0, stream>>>( \
        log_probs,                                                           \
        topk_tmp_id_buf,                                                     \
        topk_tmp_val_buf,                                                    \
        vocab_size,                                                          \
        diversity_rate);                                                     \
    if (K < 10)                                                              \
      batch_topK_kernel<                                                     \
          T,                                                                 \
          K,                                                                 \
          BLOCK_SIZE_1><<<batch_size, BLOCK_SIZE_1, 0, stream>>>(            \
          topk_tmp_id_buf, topk_tmp_val_buf, ids);                           \
    else                                                                     \
      batch_topK_kernel_v2<T, K, 32><<<batch_size, 32, 0, stream>>>(         \
          topk_tmp_id_buf, topk_tmp_val_buf, ids);                           \
    break;

#define CASE_K(K, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_)            \
  case K:                                                                    \
    topk_stage_1_opt3<float,                                                 \
                      BLOCK_SIZE_1_,                                         \
                      BLOCKS_PER_BEAM_><<<batch_size * K * BLOCKS_PER_BEAM_, \
                                          BLOCK_SIZE_1_,                     \
                                          0,                                 \
                                          stream>>>(log_probs,               \
                                                    temp_log_probs,          \
                                                    topk_tmp_id_buf,         \
                                                    topk_tmp_val_buf,        \
                                                    finished,                \
                                                    beam_width,              \
                                                    vocab_size,              \
                                                    end_id);                 \
    topk_stage_2_opt3<float,                                                 \
                      BLOCK_SIZE_2_,                                         \
                      BLOCKS_PER_BEAM_><<<batch_size,                        \
                                          BLOCK_SIZE_2_,                     \
                                          K * sizeof(int),                   \
                                          stream>>>(                         \
        topk_tmp_id_buf, topk_tmp_val_buf, ids, beam_width);                 \
    break;

template <typename T>
void topK_kernelLauncher(void* workspace,
                         size_t& workspace_size,
                         T* log_probs,
                         int* ids,
                         const bool* finished,
                         DecodingBeamsearchArguments args,
                         cudaStream_t stream) {
  const int batch_size = args.batch_size_;
  const int beam_width = args.beam_width_;
  const int vocab_size = args.vocab_size_padded_;
  const T diversity_rate = args.beam_search_diversity_rate_;
  const int end_id = args.end_id_;

  const int max_block_per_beam = 8;
  int temp_log_probs_buf_size =
      batch_size * beam_width * vocab_size;  // type float
  int topk_tmp_ids_buf_size =
      batch_size * beam_width * beam_width * max_block_per_beam;  // type int
  int topk_tmp_val_buf_size =
      batch_size * beam_width * beam_width * max_block_per_beam;  // type float
  // int can_score_buf_size = batch_size * beam_width * vocab_size;
  // int can_idx_buf_size = batch_size * beam_width * vocab_size;

  // prevent memory misalinged address
  temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
  // can_score_buf_size = (int)(ceil(can_score_buf_size / 4.)) * 4;
  // can_idx_buf_size = (int)(ceil(can_idx_buf_size / 4.)) * 4;
  topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
  topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

  if (workspace == nullptr) {
    workspace_size = sizeof(float) * temp_log_probs_buf_size +
                     sizeof(int) * topk_tmp_ids_buf_size +
                     sizeof(float) * topk_tmp_val_buf_size;
    // sizeof(float) * can_score_buf_size +
    // sizeof(int) * can_idx_buf_size;
    return;
  } else {
    T* temp_log_probs = (T*)workspace;
    int* topk_tmp_id_buf = (int*)(temp_log_probs + temp_log_probs_buf_size);
    T* topk_tmp_val_buf = (T*)(topk_tmp_id_buf + topk_tmp_ids_buf_size);
    // T* can_score_buf = (T*)(topk_tmp_val_buf + topk_tmp_val_buf_size);
    // int* can_idx_buf = (int*)(can_score_buf + can_score_buf_size);
    if (diversity_rate == 0.0f) {
      switch (beam_width) {
        CASE_K(1, 128, 128, 8);
        CASE_K(4, 128, 128, 8);
        CASE_K(10, 128, 128, 8);
        CASE_K(16, 128, 128, 5);
        CASE_K(32, 256, 128, 1);
        CASE_K(64, 256, 256, 1);
        default:
          topk_stage_1_opt2_general<
              T,
              128,
              1><<<batch_size * beam_width * 1, 128, 0, stream>>>(
              log_probs,
              temp_log_probs,
              topk_tmp_id_buf,
              topk_tmp_val_buf,
              beam_width,
              vocab_size);
          topk_stage_2_opt2_general<T, 128, 1><<<batch_size,
                                                 128,
                                                 beam_width * beam_width * 1 *
                                                         sizeof(float) +
                                                     beam_width * sizeof(int),
                                                 stream>>>(
              topk_tmp_id_buf, topk_tmp_val_buf, ids, beam_width);
          break;
      }
    } else {
      switch (beam_width) {
        CASE_K_DIV(1, 256, 256);
        CASE_K_DIV(4, 256, 256);
        CASE_K_DIV(16, 256, 64);
        CASE_K_DIV(64, 256, 64);
        default:
          // printf("[ERROR] Topk kernel does not support beamwidth = %d \n",
          //        beam_width);
          // exit(0);
          // diversity_rate only works when BLOCKS_PER_BEAM_ is 1
          topk_stage_1_opt3<T,
                            128,
                            1><<<batch_size * beam_width * 1, 128, 0, stream>>>(
              log_probs,
              (float*)nullptr,
              temp_log_probs,
              topk_tmp_id_buf,
              topk_tmp_val_buf,
              finished,
              beam_width,
              vocab_size,
              diversity_rate,
              end_id);
          topk_stage_2_opt3<
              T,
              128,
              1><<<batch_size, 128, beam_width * sizeof(int), stream>>>(
              topk_tmp_id_buf, topk_tmp_val_buf, ids, beam_width);
          break;
      }
    }
    return;
  }
}

#undef CASE_K
#undef CASE_K_DIV

template void topK_kernelLauncher<float>(void* workspace,
                                         size_t& workspace_size,
                                         float* log_probs,
                                         int* ids,
                                         const bool* finished,
                                         DecodingBeamsearchArguments args,
                                         cudaStream_t stream);

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_2_opt3_update(const int* __restrict topk_tmp_id_buf,
                                         T* topk_tmp_val_buf,
                                         bool* finished,
                                         bool* alive_finished,
                                         int* sequence_length,
                                         int* word_ids,
                                         int* parent_ids,
                                         int* output_word_ids,
                                         int* output_parent_ids,
                                         float* output_cum_log_probs,
                                         const int beam_width,
                                         const int vocab_size,
                                         const int end_id,
                                         const int step,
                                         const int max_out_len,
                                         int k,
                                         //  T diversity_rate,
                                         float length_penalty,
                                         float max_length_penalty,
                                         const int finished_candidate_num,
                                         const bool early_stopping) {
  const int size = beam_width * BLOCKS_PER_BEAM_ * beam_width * 2;
  const int tid = threadIdx.x;
  const int batch_id = blockIdx.x;
  const bool IS_FP16 = std::is_same<T, half>::value;
  // to be consistent with MAX_T_VAL in init_kernel, which should also be same
  // with other topk kernel, however it does not
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : 1e20f;

  typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  TopK_2<T> partial;

  int finish_num = 0;
  int alive_num = 0;
  // No need for tmp_cum_log_probs anymore, since topk_tmp_val_buf stores cum
  // log
  // probs now thus only need to write and no need to read output_cum_log_probs.
  // float* tmp_cum_log_probs =
  //     (float*)(topk_tmp_val_buf +
  //              gridDim.x * beam_width * beam_width * 2 * BLOCKS_PER_BEAM_) +
  //     batch_id * beam_width;
  topk_tmp_id_buf += batch_id * size;
  topk_tmp_val_buf += batch_id * size;
  word_ids += batch_id * beam_width;
  parent_ids += batch_id * beam_width;
  output_word_ids += batch_id * k;  // k == beam_width*2
  output_parent_ids += batch_id * k;
  output_cum_log_probs += batch_id * k;
  sequence_length += batch_id * k;
  finished += batch_id * k;
  alive_finished += batch_id * beam_width;

  for (int ite = 0; ite < k; ite++) {
    partial.init();
#pragma unroll
    for (int i = tid; i < size; i += BLOCK_SIZE_) {
      // diversity_rate only works when BLOCKS_PER_BEAM_ is 1
      // topk_tmp_val_buf reserves cum log probs rather than log probs currently
      // partial.insert(topk_tmp_val_buf[i] +
      //                    output_cum_log_probs[topk_tmp_id_buf[i] / vocab_size
      //                    %
      //                                             beam_width +
      //                                         beam_width] +
      //                    i % k * diversity_rate,
      //                i);
      // partial.insert(topk_tmp_val_buf[i] + i % k * diversity_rate, i);
      partial.insert(topk_tmp_val_buf[i], i);
    }

    TopK_2<T> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

    if (tid == 0) {
      if (ite == 0) {
        if (step == 1) {  // init output
          for (int i = 0; i < beam_width; i++) {
            output_word_ids[i] = end_id;
            output_cum_log_probs[i] = -1e20f;
            output_parent_ids[i] = -1;
            sequence_length[i] = 0;
          }
        } else {
          for (int i = 0; i < beam_width; i++) {
            output_word_ids[i] = end_id;
            output_parent_ids[i] = i;
            if (finished[i]) finish_num++;
          }
        }
      }

      // beam_online_softmax_topk_kernel produces absolute id, which can make
      // update_KV_cache_kernel use gather instead of gather_nd
      int abs_id = topk_tmp_id_buf[total.p];
      // use scores in topk_tmp_val_buf rather than total.u, since the latter
      // stores diversity decay values, while original cum log probs is needed.
      // float cum_log_prob = total.u;
      float cum_log_prob = (float)topk_tmp_val_buf[total.p];
      // There are two queues, one for the alive and another for the finish.
      // `beam_id` stands for parents in the alive, and it uses absolute id
      // represented as `batch_idx * beam_width + beam_idx`.
      int beam_id = abs_id / vocab_size;
      int beam_id_in_output =
          batch_id * k + (beam_id % beam_width) + beam_width;
      int word_id = abs_id % vocab_size;
      if (word_id == end_id && ite < finished_candidate_num) {
        // grow finish
        float score = cum_log_prob / length_penalty;
        if (score > output_cum_log_probs[beam_width - 1]) {
          output_word_ids[beam_width - 1] = end_id;
          output_cum_log_probs[beam_width - 1] = score;
          output_parent_ids[beam_width - 1] = beam_id_in_output;
          sequence_length[beam_width - 1] = step;
          finished[beam_width - 1] = 1;
          for (int i = beam_width - 2; i >= 0; --i) {
            if (output_cum_log_probs[i + 1] > output_cum_log_probs[i]) {
              // output_word_ids[i] = end_id;
              float tmp_f = output_cum_log_probs[i];
              output_cum_log_probs[i] = output_cum_log_probs[i + 1];
              output_cum_log_probs[i + 1] = tmp_f;
              int tmp_i = output_parent_ids[i];
              output_parent_ids[i] = output_parent_ids[i + 1];
              output_parent_ids[i + 1] = tmp_i;
              tmp_i = sequence_length[i];
              sequence_length[i] = sequence_length[i + 1];
              sequence_length[i + 1] = tmp_i;
              tmp_i = finished[i];
              finished[i] = finished[i + 1];
              finished[i + 1] = tmp_i;
            } else {
              break;
            }
          }
        }
        if (finish_num != beam_width) finish_num += 1;
      } else if (alive_num < beam_width && word_id != end_id) {
        // grow alive
        parent_ids[alive_num] = beam_id;
        word_ids[alive_num] = word_id;
        // Also put alive candidates after finish candidates, since output
        // must include both the finish and alive to trace full path
        output_word_ids[beam_width + alive_num] = word_id;
        output_parent_ids[beam_width + alive_num] = beam_id_in_output;
        // Must not override output_cum_log_probs since the after iters would
        // use it. We will copy tmp_cum_log_probs back to output_cum_log_probs
        // after the topk all has been selected.
        // tmp_cum_log_probs[alive_num] = cum_log_prob;
        // No need for tmp_cum_log_probs anymore.
        output_cum_log_probs[beam_width + alive_num] = cum_log_prob;
        sequence_length[beam_width + alive_num] = step;
        finished[beam_width + alive_num] = 0;
        alive_finished[alive_num] = 0;
        alive_num += 1;
      }
      topk_tmp_val_buf[total.p] = -MAX_T_VAL;
    }
    __syncthreads();
  }

  if (tid == 0) {
    // No need for tmp_cum_log_probs anymore.
    // for (int i = 0; i < beam_width; ++i) {
    //   output_cum_log_probs[beam_width + i] = tmp_cum_log_probs[i];
    // }
    // early finish
    float lowest_finish =
        finish_num == 0 ? -1e20f : output_cum_log_probs[finish_num - 1];
    // The best possible score of the most likely alive sequence
    float lower_bound =
        (float)output_cum_log_probs[beam_width] / max_length_penalty;

    if (finished_candidate_num == beam_width) {
      if (finish_num == finished_candidate_num &&
          (lowest_finish > lower_bound || early_stopping)) {
        // If early stop, also mark the alive beams finished.
        for (int i = beam_width; i < beam_width * 2; ++i) {
          finished[i] = 1;
          alive_finished[i - beam_width] = 1;
        }
      } else if (step == max_out_len) {
        // sort on finish sequences and alive sequences
        for (int ite = beam_width; ite < beam_width * 2; ++ite) {
          output_cum_log_probs[ite] =
              output_cum_log_probs[ite] / length_penalty;
          for (int i = ite - 1;
               i >= 0 && output_cum_log_probs[i + 1] > output_cum_log_probs[i];
               --i) {
            float tmp_f = output_cum_log_probs[i];
            output_cum_log_probs[i] = output_cum_log_probs[i + 1];
            output_cum_log_probs[i + 1] = tmp_f;
            int tmp_i = output_word_ids[i];
            output_word_ids[i] = output_word_ids[i + 1];
            output_word_ids[i + 1] = tmp_i;
            tmp_i = output_parent_ids[i];
            output_parent_ids[i] = output_parent_ids[i + 1];
            output_parent_ids[i + 1] = tmp_i;
            tmp_i = sequence_length[i];
            sequence_length[i] = sequence_length[i + 1];
            sequence_length[i + 1] = tmp_i;
            finished[i] = 1;
            finished[i + 1] = 1;
          }
        }
        // If early stop, also mark the alive beams finished.
        for (int i = beam_width; i < beam_width * 2; ++i) {
          finished[i] = 1;
          alive_finished[i - beam_width] = 1;
        }
      }

    } else {
      // output must include both the finish and alive to trace full path
      if (step == max_out_len ||
          lowest_finish > lower_bound) {  // when finishing
        for (int i = 0; finish_num < beam_width; ++finish_num, ++i) {
          output_word_ids[finish_num] = word_ids[i];
          output_cum_log_probs[finish_num] =
              output_cum_log_probs[i + beam_width] / length_penalty;
          output_parent_ids[finish_num] = output_parent_ids[i + beam_width];
          sequence_length[finish_num] = step;
          finished[finish_num] = 1;
        }
        // If early stop, also mark the alive beams finished.
        for (int i = beam_width; i < beam_width * 2; ++i) {
          finished[i] = 1;
          alive_finished[i - beam_width] = 1;
        }
      }
    }
  }
}

#define CASE_K(K, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_)            \
  case K:                                                                    \
    topk_stage_1_opt3<T,                                                     \
                      BLOCK_SIZE_1_,                                         \
                      BLOCKS_PER_BEAM_><<<batch_size * K * BLOCKS_PER_BEAM_, \
                                          BLOCK_SIZE_1_,                     \
                                          0,                                 \
                                          stream>>>(log_probs,               \
                                                    output_cum_log_probs,    \
                                                    temp_log_probs,          \
                                                    topk_tmp_id_buf,         \
                                                    topk_tmp_val_buf,        \
                                                    finished,                \
                                                    beam_width * 2,          \
                                                    vocab_size,              \
                                                    (T)diversity_rate,       \
                                                    end_id);                 \
    topk_stage_2_opt3_update<                                                \
        T,                                                                   \
        BLOCK_SIZE_2_,                                                       \
        BLOCKS_PER_BEAM_><<<batch_size, BLOCK_SIZE_2_, 0, stream>>>(         \
        topk_tmp_id_buf,                                                     \
        topk_tmp_val_buf,                                                    \
        finished,                                                            \
        alive_finished,                                                      \
        sequence_length,                                                     \
        word_ids,                                                            \
        parent_ids,                                                          \
        output_word_ids,                                                     \
        output_parent_ids,                                                   \
        output_cum_log_probs,                                                \
        beam_width,                                                          \
        vocab_size,                                                          \
        end_id,                                                              \
        step,                                                                \
        max_out_len,                                                         \
        beam_width * 2,                                                      \
        length_penalty,                                                      \
        max_length_penalty,                                                  \
        finished_candidate_num,                                              \
        early_stopping);                                                     \
    break;

template <typename T>
void topK_update_kernelLauncher(
    void* workspace,
    size_t& workspace_size,
    const T* log_probs,
    bool* finished,
    bool* alive_finished,
    int* sequence_length,
    int* word_ids,
    int* parent_ids,  // for update cache, only include alive beams
    int* output_word_ids,
    int* output_parent_ids,  // for gather tree, include both alive and finish
                             // beams
    float* output_cum_log_probs,  // NOTE: cum_log_probs is T in V3.1
    const int step,
    DecodingBeamsearchArguments args,
    cudaStream_t stream) {
  const int batch_size = args.batch_size_;
  const int beam_width = args.beam_width_;
  const int vocab_size = args.vocab_size_padded_;
  // const int vocab_size = args.vocab_size_;
  const float diversity_rate = args.beam_search_diversity_rate_;
  const int end_id = args.end_id_;
  const int max_out_len = args.seq_len_;
  const float alpha = args.alpha_;

  const int max_block_per_beam = 8;
  int temp_log_probs_buf_size =
      batch_size * beam_width * vocab_size;  // type float
  // select top beam_width*2 for topk_tmp_id_buf and topk_tmp_val_buf
  int topk_tmp_ids_buf_size = batch_size * beam_width * beam_width * 2 *
                              max_block_per_beam;  // type int
  int topk_tmp_val_buf_size = batch_size * beam_width * beam_width * 2 *
                              max_block_per_beam;  // type float
  // // to save tmp output_cum_log_probs results of the alive beams
  // topk_tmp_val_buf_size += batch_size * beam_width;

  // prevent memory misalinged address
  temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
  topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
  topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

  if (workspace == nullptr) {
    workspace_size = sizeof(float) * temp_log_probs_buf_size +
                     sizeof(int) * topk_tmp_ids_buf_size +
                     sizeof(float) * topk_tmp_val_buf_size;
    return;
  } else {
    T* temp_log_probs = (T*)workspace;
    int* topk_tmp_id_buf = (int*)(temp_log_probs + temp_log_probs_buf_size);
    T* topk_tmp_val_buf = (T*)(topk_tmp_id_buf + topk_tmp_ids_buf_size);
    const int finished_candidate_num = args.finished_candidate_num_;
    const bool early_stopping = args.early_stopping_;
    float length_penalty = (finished_candidate_num == beam_width)
                               ? std::pow((5. + step - 1) / 6., alpha)
                               : std::pow((5. + step + 1) / 6., alpha);
    float max_length_penalty =
        (finished_candidate_num == beam_width)
            ? length_penalty
            : std::pow((5. + max_out_len + 1) / 6., alpha);
    if (diversity_rate == 0.0f) {
      switch (beam_width) {
        CASE_K(1, 128, 128, 8);
        CASE_K(4, 128, 128, 8);
        CASE_K(10, 128, 128, 8);
        CASE_K(16, 128, 128, 5);
        CASE_K(32, 256, 128, 1);
        CASE_K(64, 256, 256, 1);
        default:
          topk_stage_1_opt3<T,
                            128,
                            1><<<batch_size * beam_width * 1, 128, 0, stream>>>(
              log_probs,
              output_cum_log_probs,
              temp_log_probs,
              topk_tmp_id_buf,
              topk_tmp_val_buf,
              finished,
              beam_width * 2,
              vocab_size,
              diversity_rate,
              end_id);
          topk_stage_2_opt3_update<T, 128, 1><<<batch_size, 128, 0, stream>>>(
              topk_tmp_id_buf,
              topk_tmp_val_buf,
              finished,
              alive_finished,
              sequence_length,
              word_ids,
              parent_ids,
              output_word_ids,
              output_parent_ids,
              output_cum_log_probs,
              beam_width,
              vocab_size,
              end_id,
              step,
              max_out_len,
              beam_width * 2,
              // diversity_rate,
              length_penalty,
              max_length_penalty,
              finished_candidate_num,
              early_stopping);
          break;
      }
    } else {
      // diversity_rate only works when BLOCKS_PER_BEAM_ is 1 to get the correct
      // branch indice by `idx%k`
      topk_stage_1_opt3<T,
                        128,
                        1><<<batch_size * beam_width * 1, 128, 0, stream>>>(
          log_probs,
          output_cum_log_probs,
          temp_log_probs,
          topk_tmp_id_buf,
          topk_tmp_val_buf,
          finished,
          beam_width * 2,
          vocab_size,
          diversity_rate,
          end_id);
      topk_stage_2_opt3_update<T, 128, 1><<<batch_size, 128, 0, stream>>>(
          topk_tmp_id_buf,
          topk_tmp_val_buf,
          finished,
          alive_finished,
          sequence_length,
          word_ids,
          parent_ids,
          output_word_ids,
          output_parent_ids,
          output_cum_log_probs,
          beam_width,
          vocab_size,
          end_id,
          step,
          max_out_len,
          beam_width * 2,
          // diversity_rate,
          length_penalty,
          max_length_penalty,
          finished_candidate_num,
          early_stopping);
    }
    return;
  }
}

#undef CASE_K
#undef CASE_K_DIV

template void topK_update_kernelLauncher<float>(
    void* workspace,
    size_t& workspace_size,
    const float* log_probs,
    bool* finished,
    bool* alive_finished,
    int* sequence_length,
    int* word_ids,
    int* parent_ids,  // for update cache, only include alive beams
    int* output_word_ids,
    int* output_parent_ids,  // for gather tree, include both alive and finish
                             // beams
    float* output_cum_log_probs,  // NOTE: cum_log_probs is T in V3.1
    const int step,
    DecodingBeamsearchArguments args,
    cudaStream_t stream);

template void topK_update_kernelLauncher<half>(
    void* workspace,
    size_t& workspace_size,
    const half* log_probs,
    bool* finished,
    bool* alive_finished,
    int* sequence_length,
    int* word_ids,
    int* parent_ids,  // for update cache, only include alive beams
    int* output_word_ids,
    int* output_parent_ids,  // for gather tree, include both alive and finish
                             // beams
    float* output_cum_log_probs,  // NOTE: cum_log_probs is T in V3.1
    const int step,
    DecodingBeamsearchArguments args,
    cudaStream_t stream);

// Sampling kernels
template <typename T>
__global__ void sampling(int* topk_tmp_id_buf,
                         T* topk_tmp_val_buf,
                         int* ids,
                         int* sequence_length,
                         bool* finished_buf,
                         const int candidate_num,
                         int random_num,
                         const int end_id,
                         const int vocab_size) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ float sum;
  __shared__ float rand_num;

  if (tid < candidate_num) {
    float max_val = topk_tmp_val_buf[bid * candidate_num];
    topk_tmp_val_buf[bid * candidate_num + tid] =
        (T)__expf((float)topk_tmp_val_buf[bid * candidate_num + tid] - max_val);
  }

  if (tid == 0) {
    sum = 0.0f;
    for (int i = 0; i < candidate_num; i++) {
      sum = sum + (float)topk_tmp_val_buf[bid * candidate_num + i];
    }

    curandState_t local_state;
    curand_init(
        (T)0, bid * candidate_num, blockDim.x * candidate_num, &local_state);
    rand_num = (float)curand_uniform(&local_state) * sum;

    ids[bid] =
        topk_tmp_id_buf[bid * candidate_num + candidate_num - 1] % vocab_size;
    for (int i = 0; i < candidate_num; i++) {
      rand_num = rand_num - (float)topk_tmp_val_buf[bid * candidate_num + i];
      if (rand_num <= 0.0f) {
        ids[bid] = topk_tmp_id_buf[bid * candidate_num + i] % vocab_size;
        break;
      }
    }
    if (finished_buf != nullptr) {
      if (sequence_length != nullptr) {
        sequence_length[bid] =
            finished_buf[bid] ? sequence_length[bid] : sequence_length[bid] + 1;
      }
      finished_buf[bid] = ids[bid] == end_id ? 1 : 0;
    }
  }
}

#define CASE_K(K)                                                              \
  case K:                                                                      \
    beam_topK_kernel<T, K, block_size><<<batch_size, block_size, 0, stream>>>( \
        log_probs, topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, 0.0f);       \
    break;

template <typename T>
void topK_sampling_kernel_kernelLauncher(void* workspace,
                                         size_t& workspace_size,
                                         T* log_probs,
                                         int* ids,
                                         int* sequence_length,
                                         bool* finished_buf,
                                         int random_num,
                                         DecodingSamplingArguments args,
                                         cudaStream_t stream,
                                         const int batch_size) {
  // This function would be called two or more times.
  // First time is used to get the workspace size, so we need to put
  // max batch size we want to use.
  // For other times, we need to put the inference batch size to
  // set the grid size we use.
  const int vocab_size = args.vocab_size_padded_;
  const int candidate_num = args.candidate_num_;
  const int end_id = args.end_id_;
  const int block_size = 256;

  int topk_tmp_ids_buf_size =
      args.batch_size_ * args.candidate_num_;  // type int
  int temp_log_probs_buf_size = args.batch_size_ * vocab_size;
  int topk_tmp_val_buf_size = args.batch_size_ * args.candidate_num_;  // type T

  temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
  topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
  topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

  if (workspace == nullptr) {
    workspace_size = sizeof(T) * temp_log_probs_buf_size +
                     sizeof(int) * topk_tmp_ids_buf_size +
                     sizeof(T) * topk_tmp_val_buf_size;
  } else {
    T* temp_log_probs = (T*)workspace;
    int* topk_tmp_id_buf = (int*)(temp_log_probs + temp_log_probs_buf_size);
    T* topk_tmp_val_buf = (T*)(topk_tmp_id_buf + topk_tmp_ids_buf_size);

    switch (candidate_num) {
      CASE_K(1);
      CASE_K(2);
      CASE_K(4);
      CASE_K(16);
      CASE_K(64);
      default:
        beam_topK_kernel_general<
            T,
            block_size><<<batch_size, block_size, 0, stream>>>(log_probs,
                                                               temp_log_probs,
                                                               topk_tmp_id_buf,
                                                               topk_tmp_val_buf,
                                                               candidate_num,
                                                               vocab_size);
        break;
    }
    sampling<T><<<batch_size, candidate_num, 0, stream>>>(topk_tmp_id_buf,
                                                          topk_tmp_val_buf,
                                                          ids,
                                                          sequence_length,
                                                          finished_buf,
                                                          candidate_num,
                                                          random_num,
                                                          end_id,
                                                          vocab_size);
  }
}

#undef CASE_K

#define CASE_K(K_MIN, K_MAX, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_) \
  case K_MIN... K_MAX:                                                       \
    topk_stage_1_opt3<T,                                                     \
                      BLOCK_SIZE_1_,                                         \
                      BLOCKS_PER_BEAM_><<<batch_size * BLOCKS_PER_BEAM_,     \
                                          BLOCK_SIZE_1_,                     \
                                          0,                                 \
                                          stream>>>(log_probs,               \
                                                    temp_log_probs,          \
                                                    topk_tmp_id_buf,         \
                                                    topk_tmp_val_buf,        \
                                                    finished_buf,            \
                                                    candidate_num,           \
                                                    vocab_size,              \
                                                    end_id);                 \
    topk_stage_2_opt3_sampling<T,                                            \
                               BLOCK_SIZE_2_,                                \
                               BLOCKS_PER_BEAM_><<<batch_size,               \
                                                   BLOCK_SIZE_2_,            \
                                                   K_MAX * sizeof(int),      \
                                                   stream>>>(                \
        topk_tmp_id_buf,                                                     \
        topk_tmp_val_buf,                                                    \
        topk_tmp2_val_buf,                                                   \
        ids,                                                                 \
        sequence_length,                                                     \
        finished_buf,                                                        \
        candidate_num,                                                       \
        curandstate,                                                         \
        end_id,                                                              \
        vocab_size);                                                         \
    break;


template <typename T>
void topK_sampling_kernel_kernelLauncher_v2(void* workspace,
                                            size_t& workspace_size,
                                            T* log_probs,
                                            int* ids,
                                            int* sequence_length,
                                            bool* finished_buf,
                                            curandState_t* curandstate,
                                            DecodingSamplingArguments args,
                                            cudaStream_t stream,
                                            const int batch_size) {
  // Here, we put batch size as an argument because the batch size of
  // initialization
  // and inference may be different due to pipelint parallelism.
  const int candidate_num = args.candidate_num_;
  const int vocab_size = args.vocab_size_padded_;
  const int end_id = args.end_id_;

  const int max_block_per_beam = 8;
  int temp_log_probs_buf_size = batch_size * vocab_size;  // type float
  int topk_tmp_ids_buf_size =
      batch_size * candidate_num * max_block_per_beam;  // type int
  int topk_tmp_val_buf_size =
      batch_size * candidate_num * max_block_per_beam;  // type float

  // prevent memory misalinged address
  temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
  topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
  topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

  if (workspace == nullptr) {
    workspace_size = sizeof(T) * temp_log_probs_buf_size +
                     sizeof(int) * topk_tmp_ids_buf_size +
                     2 * sizeof(T) * topk_tmp_val_buf_size;
    return;
  } else {
    T* temp_log_probs = (T*)workspace;
    int* topk_tmp_id_buf = (int*)(temp_log_probs + temp_log_probs_buf_size);
    T* topk_tmp_val_buf = (T*)(topk_tmp_id_buf + topk_tmp_ids_buf_size);
    T* topk_tmp2_val_buf = (T*)(topk_tmp_val_buf + topk_tmp_val_buf_size);

    switch (candidate_num) {
      CASE_K(1, 16, 128, 128, 8);
      CASE_K(17, 32, 256, 128, 8);
      CASE_K(33, 64, 256, 256, 8);
      default:
        printf("[ERROR] Topk kernel does not support candidate_num = %d \n",
               candidate_num);
        exit(0);
        break;
    }
    return;
  }
}

#undef CASE_K


template void topK_sampling_kernel_kernelLauncher(
    void* workspace,
    size_t& workspace_size,
    float* log_probs,
    int* ids,
    int* sequence_length,
    bool* finished_buf,
    int random_num,
    DecodingSamplingArguments args,
    cudaStream_t stream,
    const int batch_size);

template void topK_sampling_kernel_kernelLauncher(
    void* workspace,
    size_t& workspace_size,
    half* log_probs,
    int* ids,
    int* sequence_length,
    bool* finished_buf,
    int random_num,
    DecodingSamplingArguments args,
    cudaStream_t stream,
    const int batch_size);

template void topK_sampling_kernel_kernelLauncher_v2(
    void* workspace,
    size_t& workspace_size,
    float* log_probs,
    int* ids,
    int* sequence_length,
    bool* finished_buf,
    curandState_t* curandstate,
    DecodingSamplingArguments args,
    cudaStream_t stream,
    const int batch_size);

template void topK_sampling_kernel_kernelLauncher_v2(
    void* workspace,
    size_t& workspace_size,
    half* log_probs,
    int* ids,
    int* sequence_length,
    bool* finished_buf,
    curandState_t* curandstate,
    DecodingSamplingArguments args,
    cudaStream_t stream,
    const int batch_size);


__global__ void init_topp_id_val(int* topp_id_val_buf,
                                 int* topp_offset_buf,
                                 const int batch_size,
                                 const int vocab_size) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (bid == 0) {
    for (int i = tid; i < batch_size + 1; i += blockDim.x) {
      topp_offset_buf[i] = i * vocab_size;
    }
  }

  while (tid < vocab_size) {
    topp_id_val_buf[bid * vocab_size + tid] = tid;
    tid += blockDim.x;
  }
}


void init_topp_id_val_kernel_kernelLauncher(int* topp_id_val_buf,
                                            int* topp_offset_buf,
                                            const int batch_size,
                                            const int vocab_size,
                                            cudaStream_t stream) {
  init_topp_id_val<<<batch_size, 512, 0, stream>>>(
      topp_id_val_buf, topp_offset_buf, batch_size, vocab_size);
}

// Sampling kernels
template <typename T>
__global__ void top_p_sampling(T* sorted_log_probs,
                               int* sorted_id_vals,
                               int* ids,
                               int* sequence_length,
                               bool* finished_buf,
                               const int vocab_size,
                               const int random_num,
                               const float prob_threshold,
                               const int end_id) {
  int tid = threadIdx.x;
  curandState_t local_state;
  // TODO: fix randomly cannot work in some specific situation.
  curand_init((T)random_num, tid, 0, &local_state);
  T rand_num = (T)curand_uniform(&local_state) * (T)prob_threshold;
  ids[tid] = sorted_id_vals[tid * vocab_size];

  for (int i = tid * vocab_size; i < tid * vocab_size + vocab_size; i++) {
    rand_num = rand_num - sorted_log_probs[i];
    if (rand_num <= (T)0.0f) {
      ids[tid] = sorted_id_vals[i];
      break;
    }
  }
  if (finished_buf != nullptr) {
    finished_buf[tid] = ids[tid] == end_id ? 1 : 0;
    if (sequence_length != nullptr) {
      sequence_length[tid] =
          finished_buf[tid] ? sequence_length[tid] : sequence_length[tid] + 1;
    }
  }
}

template <typename T>
__global__ void top_p_sampling_v2(T* sorted_log_probs,
                                  int* sorted_id_vals,
                                  int* ids,
                                  int* sequence_length,
                                  bool* finished_buf,
                                  const int vocab_size,
                                  curandState_t* curandstate,
                                  const float prob_threshold,
                                  const int end_id,
                                  const int batch_size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < batch_size) {
    T rand_num = (T)curand_uniform(curandstate + tid) * (T)prob_threshold;
    ids[tid] = sorted_id_vals[vocab_size - 1];
    for (int i = tid * vocab_size; i < tid * vocab_size + vocab_size; i++) {
      rand_num = rand_num - sorted_log_probs[i];
      if (rand_num <= (T)0.0) {
        ids[tid] = sorted_id_vals[i];
        break;
      }
    };
    if (finished_buf != nullptr) {
      finished_buf[tid] = ids[tid] == end_id ? 1 : 0;
      if (sequence_length != nullptr) {
        sequence_length[tid] =
            finished_buf[tid] ? sequence_length[tid] : sequence_length[tid] + 1;
      }
    }
  }
}

template <typename T>
__global__ void sort_kernel(const T* log_probs,
                            const int* id_vals,
                            T* sorted_log_probs,
                            int* sorted_id_vals,
                            const int vocab_size) {
  typedef cub::BlockRadixSort<T, 256, 32, int> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;
  // Obtain a segment of consecutive items that are blocked across threads
  T thread_keys[32];
  int thread_values[32];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  for (int i = 0; i < 32; i++) {
    int index = tid + 256 * i + bid * vocab_size;
    thread_keys[i] = log_probs[index];
    thread_values[i] = id_vals[index];
  }
  BlockRadixSort(temp_storage).SortDescending(thread_keys, thread_values);

  for (int i = 0; i < 32; i++) {
    int index = tid + 256 * i + bid * vocab_size;
    sorted_log_probs[index] = thread_keys[i];
    sorted_id_vals[index] = thread_values[i];
  }
}

template <typename T>
void topP_sampling_kernel_kernelLauncher(void* workspace,
                                         size_t& workspace_size,
                                         const T* log_probs,
                                         const int* id_vals,
                                         const int* offset_buf,
                                         bool* finished_buf,
                                         int step,
                                         DecodingSamplingArguments& args,
                                         int* output_ids,
                                         int* sequence_length,
                                         const int n,
                                         cudaStream_t stream,
                                         const int batch_size) {
  const int vocab_size = args.vocab_size_padded_;
  int sorted_log_prob_buf_size = batch_size * vocab_size;  // type T
  int sorted_id_vals_buf_size = batch_size * vocab_size;   // type int
  sorted_log_prob_buf_size = (int)(ceil(sorted_log_prob_buf_size / 4.)) * 4;
  sorted_id_vals_buf_size = (int)(ceil(sorted_id_vals_buf_size / 4.)) * 4;

  void* cub_temp_storage = workspace;
  T* sorted_log_probs =
      (T*)((char*)cub_temp_storage + args.cub_temp_storage_size_);
  int* sorted_id_vals = (int*)(sorted_log_probs + sorted_log_prob_buf_size);

  if (workspace == nullptr) {
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr,
        args.cub_temp_storage_size_,
        log_probs,
        (T*)nullptr,
        id_vals,
        (int*)nullptr,
        vocab_size * batch_size,
        batch_size,
        offset_buf,
        offset_buf + 1,
        0,              // begin_bit
        sizeof(T) * 8,  // end_bit = sizeof(KeyT) * 8
        stream);        // cudaStream_t
    args.cub_temp_storage_size_ =
        (int)(ceil(args.cub_temp_storage_size_ / 4.)) * 4;
    workspace_size = sizeof(T) * sorted_log_prob_buf_size +
                     sizeof(int) * sorted_id_vals_buf_size +
                     args.cub_temp_storage_size_;
  } else {
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        cub_temp_storage,
        args.cub_temp_storage_size_,
        log_probs,
        sorted_log_probs,
        id_vals,
        sorted_id_vals,
        n * batch_size,
        batch_size,
        offset_buf,
        offset_buf + 1,
        0,              // begin_bit
        sizeof(T) * 8,  // end_bit = sizeof(KeyT) * 8
        stream);        // cudaStream_t

    top_p_sampling<<<1, batch_size, 0, stream>>>(sorted_log_probs,
                                                 sorted_id_vals,
                                                 output_ids,
                                                 sequence_length,
                                                 finished_buf,
                                                 n,
                                                 step,
                                                 args.probability_threshold_,
                                                 args.end_id_);
  }
}

template void topP_sampling_kernel_kernelLauncher(
    void* workspace,
    size_t& workspace_size,
    const float* log_probs,
    const int* id_vals,
    const int* offset_buf,
    bool* finished_buf,
    int step,
    DecodingSamplingArguments& args,
    int* output_ids,
    int* sequence_length,
    const int n,
    cudaStream_t stream,
    const int batch_size);

template void topP_sampling_kernel_kernelLauncher(
    void* workspace,
    size_t& workspace_size,
    const half* log_probs,
    const int* id_vals,
    const int* offset_buf,
    bool* finished_buf,
    int step,
    DecodingSamplingArguments& args,
    int* output_ids,
    int* sequence_length,
    const int n,
    cudaStream_t stream,
    const int batch_size);


template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void beam_topK_kernel_for_topP(const T* log_probs,
                                   int* topk_tmp_id_buf,
                                   T* topk_tmp_val_buf,
                                   const int vocab_size,
                                   int* offset_buf,
                                   int* begin_offset_buf,
                                   float p_threshold) {
  typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  TopK<T, MAX_K> partial;

  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

#pragma unroll
  for (int i = 0; i < MAX_K; ++i) {
    partial.p[i] = -1;
    partial.u[i] = -MAX_T_VAL;
  }

#pragma unroll
  for (int elem_id = thread_id; elem_id < vocab_size;
       elem_id += THREADBLOCK_SIZE) {
    int index = elem_id + block_id * vocab_size;
    partial.insert(log_probs[index], index);
  }

  TopK<T, MAX_K> total =
      BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);


  if (thread_id == 0) {
    begin_offset_buf[block_id] = offset_buf[block_id];
    T sum_prob = (T)(0.0f);

#pragma unroll
    for (int i = 0; i < MAX_K; i++) {
      sum_prob += total.u[i];
    }

    if ((float)sum_prob >= p_threshold) {
      begin_offset_buf[block_id] += vocab_size;
      int index = block_id * vocab_size;

#pragma unroll
      for (int i = 0; i < MAX_K; ++i) {
        topk_tmp_id_buf[index + i] = total.p[i] % vocab_size;
        topk_tmp_val_buf[index + i] = total.u[i];
      }
    }
  }
}

template <typename T>
void topP_sampling_kernel_kernelLauncher_v2(void* workspace,
                                            size_t& workspace_size,
                                            const T* log_probs,
                                            const int* id_vals,
                                            int* offset_buf,
                                            int* begin_offset_buf,
                                            bool* finished_buf,
                                            curandState_t* curandstate,
                                            DecodingSamplingArguments& args,
                                            int* output_ids,
                                            int* sequence_length,
                                            const int n,
                                            cudaStream_t stream,
                                            const int batch_size) {
  // Here, we put batch size as an argument because the batch size of
  // initialization
  // and inference may be different due to pipelint parallelism.
  const int vocab_size = args.vocab_size_padded_;
  const int block_size = 256;

  int sorted_log_prob_buf_size = batch_size * vocab_size;  // type T
  int sorted_id_vals_buf_size = batch_size * vocab_size;   // type int
  sorted_log_prob_buf_size = (int)(ceil(sorted_log_prob_buf_size / 4.)) * 4;
  sorted_id_vals_buf_size = (int)(ceil(sorted_id_vals_buf_size / 4.)) * 4;

  void* cub_temp_storage = workspace;
  T* sorted_log_probs =
      (T*)((char*)cub_temp_storage + args.cub_temp_storage_size_);
  int* sorted_id_vals = (int*)(sorted_log_probs + sorted_log_prob_buf_size);


  if (workspace == nullptr) {
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr,
        args.cub_temp_storage_size_,
        log_probs,
        (T*)nullptr,
        id_vals,
        (int*)nullptr,
        vocab_size * batch_size,
        batch_size,
        begin_offset_buf,
        offset_buf + 1,
        0,              // begin_bit
        sizeof(T) * 8,  // end_bit = sizeof(KeyT) * 8
        stream);        // cudaStream_t
    args.cub_temp_storage_size_ =
        (int)(ceil(args.cub_temp_storage_size_ / 4.)) * 4;
    workspace_size = sizeof(T) * sorted_log_prob_buf_size +
                     sizeof(int) * sorted_id_vals_buf_size +
                     args.cub_temp_storage_size_;
  } else {
    beam_topK_kernel_for_topP<
        T,
        1,
        block_size><<<batch_size, block_size, 0, stream>>>(
        log_probs,
        sorted_id_vals,
        sorted_log_probs,
        vocab_size,
        offset_buf,
        begin_offset_buf,
        args.probability_threshold_);

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        cub_temp_storage,
        args.cub_temp_storage_size_,
        log_probs,
        sorted_log_probs,
        id_vals,
        sorted_id_vals,
        n * batch_size,
        batch_size,
        begin_offset_buf,
        offset_buf + 1,
        0,              // begin_bit
        sizeof(T) * 8,  // end_bit = sizeof(KeyT) * 8
        stream);        // cudaStream_t

    dim3 block(256);
    dim3 grid((int)(ceil(batch_size * 1.0 / 256)));
    top_p_sampling_v2<<<grid, block, 0, stream>>>(sorted_log_probs,
                                                  sorted_id_vals,
                                                  output_ids,
                                                  sequence_length,
                                                  finished_buf,
                                                  n,
                                                  curandstate,
                                                  args.probability_threshold_,
                                                  args.end_id_,
                                                  batch_size);
  }
}

template void topP_sampling_kernel_kernelLauncher_v2(
    void* workspace,
    size_t& workspace_size,
    const float* log_probs,
    const int* id_vals,
    int* offset_buf,
    int* begin_offset_buf,
    bool* finished_buf,
    curandState_t* curandstate,
    DecodingSamplingArguments& args,
    int* output_ids,
    int* sequence_length,
    const int n,
    cudaStream_t stream,
    const int batch_size);

template void topP_sampling_kernel_kernelLauncher_v2(
    void* workspace,
    size_t& workspace_size,
    const half* log_probs,
    const int* id_vals,
    int* offset_buf,
    int* begin_offset_buf,
    bool* finished_buf,
    curandState_t* curandstate,
    DecodingSamplingArguments& args,
    int* output_ids,
    int* sequence_length,
    const int n,
    cudaStream_t stream,
    const int batch_size);

template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void topK_topP_sampling_kernel(int* output_ids,
                                   const T* logits,
                                   const int vocab_size,
                                   const int random_num,
                                   const float prob_threshold,
                                   T diversity_rate) {
  typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  TopK<T, MAX_K> partial;

  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

#pragma unroll
  for (int i = 0; i < MAX_K; ++i) {
    partial.p[i] = -1;
    partial.u[i] = -MAX_T_VAL;
  }

#pragma unroll
  for (int elem_id = thread_id; elem_id < vocab_size;
       elem_id += THREADBLOCK_SIZE) {
    int index = elem_id + block_id * vocab_size;
    partial.insert(logits[index], index);
  }

  TopK<T, MAX_K> total =
      BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

  if (thread_id == 0) {
    // float sum = 0.0f;
    T sum = (T)(0.0f);
    T max_val = total.u[0];

#pragma unroll
    for (int i = 0; i < MAX_K; i++) {
      total.u[i] =
          total.u[i] + diversity_rate * (T)i;  // diversely sampling penalty
      total.u[i] = (T)__expf((float)(total.u[i] - max_val));
      sum += total.u[i];
    }

    curandState_t local_state;
    curand_init((T)0, block_id * MAX_K, blockDim.x * MAX_K, &local_state);
    T rand_num = (T)curand_uniform(&local_state) * (T)prob_threshold * sum;

    output_ids[block_id] = total.p[0] % vocab_size;

#pragma unroll
    for (int i = 0; i < MAX_K; i++) {
      rand_num = rand_num - total.u[i];
      if (rand_num <= (T)0.0f) {
        output_ids[block_id] = total.p[i] % vocab_size;
        break;
      }
    }
  }
}

#define CASE_K(K)                                                          \
  case K:                                                                  \
    topK_topP_sampling_kernel<                                             \
        T,                                                                 \
        K,                                                                 \
        block_size><<<batch_size, block_size, 0, stream>>>(                \
        output_ids, logits, vocab_size, random_num, prob_threshold, 0.0f); \
    break;

template <typename T>
void topK_topP_sampling_kernel_kernelLauncher(void* workspace,
                                              size_t& workspace_size,
                                              int* output_ids,
                                              const T* logits,
                                              const int random_num,
                                              DecodingSamplingArguments& args,
                                              cudaStream_t stream,
                                              const int batch_size) {
  if (workspace == nullptr) {
    workspace_size = 0;
  } else {
    const int vocab_size = args.vocab_size_padded_;
    const int block_size = 256;
    const T prob_threshold = args.probability_threshold_;
    switch (args.candidate_num_) {
      CASE_K(1);
      CASE_K(2);
      CASE_K(4);
      CASE_K(16);
      CASE_K(64);
      default:
        printf("[ERROR] Topk kernel does not support candidate_num = %d \n",
               args.candidate_num_);
        exit(0);
        break;
    }
  }
}

#undef CASE_K

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_topp_sampling_kernel_v2(
    const int* __restrict topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    T* topk_tmp2_val_buf,
    int* ids,
    int* sequence_length,
    bool* finished_buf,
    const int k,
    const T prob_threshold,
    curandState_t* curandstate,
    const int end_id,
    const int vocab_size) {
  const int size = k * BLOCKS_PER_BEAM_;
  const int tid = threadIdx.x;
  const int batch_id = blockIdx.x;
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

  typedef cub::BlockReduce<TopK_2<float>, BLOCK_SIZE_> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  extern __shared__ char array[];
  __shared__ float rand_num;
  __shared__ float s_max;
  __shared__ float s_sum;
  T* s_val = topk_tmp_val_buf + batch_id * size;
  int* s_id = (int*)(array);
  s_max = 0.0f;
  s_sum = 0.0f;
  TopK_2<float> partial;

  for (int index = tid; index < size; index += BLOCK_SIZE_) {
    topk_tmp2_val_buf[batch_id * size + index] =
        topk_tmp_val_buf[batch_id * size + index];
  }
  __syncthreads();
  T* s_val2 = topk_tmp2_val_buf + batch_id * size;

  for (int ite = 0; ite < k; ite++) {
    partial.init();
#pragma unroll
    for (int i = tid; i < size; i += BLOCK_SIZE_) {
      partial.insert((float)s_val[i], i);
    }

    TopK_2<float> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<float>);

    if (ite == 0) s_max = total.u;

    if (tid == 0) {
      s_id[ite] = total.p;
      s_val[total.p] = -MAX_T_VAL;
      total.u = __expf(total.u - s_max);
      s_val2[total.p] = (T)total.u;
      s_sum += total.u;
    }
    __syncthreads();
  }
  if (tid == 0) {
    rand_num = (float)curand_uniform(curandstate + blockIdx.x) *
               (float)prob_threshold * s_sum;
    for (int i = 0; i < k; i++) {
      rand_num = rand_num - (float)s_val2[s_id[i]];
      if (rand_num <= 0.0f || i == k - 1) {
        ids[batch_id] = topk_tmp_id_buf[batch_id * size + s_id[i]] % vocab_size;
        break;
      }
    }
    if (finished_buf != nullptr) {
      finished_buf[batch_id] = ids[batch_id] == end_id ? 1 : 0;
      if (sequence_length != nullptr) {
        sequence_length[batch_id] = finished_buf[batch_id]
                                        ? sequence_length[batch_id]
                                        : sequence_length[batch_id] + 1;
      }
    }
  }
}

#define CASE_K(K_MIN, K_MAX, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_) \
  case K_MIN... K_MAX:                                                       \
    topk_stage_1_opt3<T,                                                     \
                      BLOCK_SIZE_1_,                                         \
                      BLOCKS_PER_BEAM_><<<batch_size * BLOCKS_PER_BEAM_,     \
                                          BLOCK_SIZE_1_,                     \
                                          0,                                 \
                                          stream>>>(logits,                  \
                                                    temp_logits,             \
                                                    topk_tmp_id_buf,         \
                                                    topk_tmp_val_buf,        \
                                                    finished_buf,            \
                                                    candidate_num,           \
                                                    vocab_size,              \
                                                    end_id);                 \
    topk_topp_sampling_kernel_v2<T,                                          \
                                 BLOCK_SIZE_2_,                              \
                                 BLOCKS_PER_BEAM_><<<batch_size,             \
                                                     BLOCK_SIZE_2_,          \
                                                     K_MAX * sizeof(int),    \
                                                     stream>>>(              \
        topk_tmp_id_buf,                                                     \
        topk_tmp_val_buf,                                                    \
        topk_tmp2_val_buf,                                                   \
        output_ids,                                                          \
        nullptr,                                                             \
        finished_buf,                                                        \
        candidate_num,                                                       \
        prob_threshold,                                                      \
        curandstate,                                                         \
        end_id,                                                              \
        vocab_size);                                                         \
    break;

template <typename T>
void topK_topP_sampling_kernel_kernelLauncher_v2(
    void* workspace,
    size_t& workspace_size,
    int* output_ids,
    const T* logits,
    bool* finished_buf,
    curandState_t* curandstate,
    DecodingSamplingArguments& args,
    cudaStream_t stream,
    const int batch_size) {
  // Here, we put batch size as an argument because the batch size of
  // initialization
  // and inference may be different due to pipelint parallelism.
  const int candidate_num = args.candidate_num_;
  const int vocab_size = args.vocab_size_padded_;
  const int end_id = args.end_id_;
  const T prob_threshold = args.probability_threshold_;

  const int max_block_per_beam = 8;
  int temp_logits_buf_size = batch_size * vocab_size;  // type float
  int topk_tmp_ids_buf_size =
      batch_size * candidate_num * max_block_per_beam;  // type int
  int topk_tmp_val_buf_size =
      batch_size * candidate_num * max_block_per_beam;  // type float

  // prevent memory misalinged address
  temp_logits_buf_size = (int)(ceil(temp_logits_buf_size / 4.)) * 4;
  topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
  topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

  if (workspace == nullptr) {
    workspace_size = sizeof(T) * temp_logits_buf_size +
                     sizeof(int) * topk_tmp_ids_buf_size +
                     2 * sizeof(T) * topk_tmp_val_buf_size;
    return;
  } else {
    T* temp_logits = (T*)workspace;
    int* topk_tmp_id_buf = (int*)(temp_logits + temp_logits_buf_size);
    T* topk_tmp_val_buf = (T*)(topk_tmp_id_buf + topk_tmp_ids_buf_size);
    T* topk_tmp2_val_buf = (T*)(topk_tmp_val_buf + topk_tmp_val_buf_size);

    switch (candidate_num) {
      CASE_K(1, 16, 128, 128, 8);
      CASE_K(17, 32, 256, 128, 8);
      CASE_K(33, 64, 256, 256, 8);
      default:
        printf("[ERROR] Topk kernel does not support candidate_num = %d \n",
               candidate_num);
        exit(0);
        break;
    }
    return;
  }
}

#undef CASE_K

template void topK_topP_sampling_kernel_kernelLauncher(
    void* workspace,
    size_t& workspace_size,
    int* output_ids,
    const float* logits,
    const int random_num,
    DecodingSamplingArguments& args,
    cudaStream_t stream,
    const int batch_size);


template void topK_topP_sampling_kernel_kernelLauncher(
    void* workspace,
    size_t& workspace_size,
    int* output_ids,
    const half* logits,
    const int random_num,
    DecodingSamplingArguments& args,
    cudaStream_t stream,
    const int batch_size);

template void topK_topP_sampling_kernel_kernelLauncher_v2(
    void* workspace,
    size_t& workspace_size,
    int* output_ids,
    const float* logits,
    bool* finished_buf,
    curandState_t* curandstate,
    DecodingSamplingArguments& args,
    cudaStream_t stream,
    const int batch_size);

template void topK_topP_sampling_kernel_kernelLauncher_v2(
    void* workspace,
    size_t& workspace_size,
    int* output_ids,
    const half* logits,
    bool* finished_buf,
    curandState_t* curandstate,
    DecodingSamplingArguments& args,
    cudaStream_t stream,
    const int batch_size);

}  // end of namespace fastertransformer
