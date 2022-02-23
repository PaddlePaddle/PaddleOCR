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

#include "cub/cub.cuh"
#include "fastertransformer/cuda/topk_kernels.cuh"

namespace fastertransformer {

#define TOPK_FP16_STORAGE 0

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

  for (int i = 0; i < MAX_K; ++i) {
    partial.p[i] = -1;
    partial.u[i] = -FLT_MAX;
  }

  for (int elem_id = thread_id; elem_id < vocab_size;
       elem_id += THREADBLOCK_SIZE) {
    int index = elem_id + block_id * vocab_size;
    partial.insert((T)log_probs[index], index);
  }

  TopK<T, MAX_K> total =
      BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

  if (thread_id == 0) {
    int index = block_id * MAX_K;

    for (int i = 0; i < MAX_K; ++i) {
      topk_tmp_id_buf[index + i] = total.p[i];
      topk_tmp_val_buf[index + i] = total.u[i] + diversity_rate * (T)i;
    }
  }
}

template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void batch_topK_kernel(int* topk_tmp_id_buf,
                           T* topk_tmp_val_buf,
                           int* id_buf) {
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  TopK<T, MAX_K> partial;
  if (thread_id == 0) {
    for (int i = 0; i < MAX_K; ++i) {
      partial.p[i] = -1;
      partial.u[i] = -FLT_MAX;
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
    void batch_topK_kernel(const int* __restrict topk_tmp_id_buf,
                           const T* __restrict topk_tmp_val_buf,
                           int* __restrict id_buf,
                           T* __restrict val_buf) {
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  TopK<T, MAX_K> partial;
  if (thread_id == 0) {
    for (int i = 0; i < MAX_K; ++i) {
      partial.p[i] = -1;
      partial.u[i] = -FLT_MAX;
    }

    int index = block_id * MAX_K * MAX_K;
    for (int i = 0; i < MAX_K * MAX_K; i++) {
      partial.insert((T)topk_tmp_val_buf[index + i],
                     topk_tmp_id_buf[index + i]);
    }

    index = block_id * MAX_K;
    for (int i = 0; i < MAX_K; i++) {
      id_buf[index + i] = partial.p[i];
      val_buf[index + i] = partial.u[i];
    }
  }
}

template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void batch_topk_kernel(const int* __restrict x,
                           const T* __restrict y,
                           int* __restrict z,
                           float* __restrict v,
                           int V,
                           int K,
                           T diversity_rate) {
  int thread_id = threadIdx.x;
  int vector_id = blockIdx.x;

  // reposition x, y to data for the current vector
  x += vector_id * V;
  y += vector_id * V;

  typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  TopK<T, MAX_K> partial;
  for (int i = 0; i < MAX_K; ++i) {
    partial.p[i] = -1;
    partial.u[i] = -FLT_MAX;
  }
  for (int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE) {
    int i = elem_id % K;
    T elem = y[elem_id] + diversity_rate * (T)i;
    int elem_idx = elem_id;  // x[elem_id];
    partial.insert(elem, elem_idx);
  }

  TopK<T, MAX_K> total =
      BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

  if (thread_id == 0) {
    z += vector_id * K;
    v += vector_id * K;

    for (int i = 0; i < MAX_K; ++i) {
      if (i < K) {
        z[i] = x[total.p[i]];
        v[i] = (float)y[total.p[i]];
      }
    }
  }
}

struct __align__(8) MD {
  float m;
  float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b) {
  bool a_bigger = (a.m > b.m);
  MD bigger_m = a_bigger ? a : b;
  MD smaller_m = a_bigger ? b : a;
  MD res;
  res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
  res.m = bigger_m.m;
  return res;
}

template <typename T, int MAX_K>
struct TopKMD {
  MD md;
  TopK<T, MAX_K> topk;
};

template <typename T, int MAX_K>
__device__ __forceinline__ TopKMD<T, MAX_K> reduce_topk_md_op(
    const TopKMD<T, MAX_K>& a, const TopKMD<T, MAX_K>& b) {
  TopKMD<T, MAX_K> res;
  res.md = reduce_md_op(a.md, b.md);
  res.topk = reduce_topk_op(a.topk, b.topk);
  return res;
}

template <typename T,
          int ITEMS_PER_THREAD,
          int MAX_K,
          int THREADBLOCK_SIZE,
          bool ALIVE = false>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void beam_online_softmax_topk_kernel(const T* __restrict x,
                                         const T* __restrict b,
                                         const float* __restrict c,
                                         const bool* __restrict finished,
                                         int* __restrict z,
                                         T* __restrict v,
                                         int V,
                                         int K,
                                         int E) {
  int thread_id = threadIdx.x;
  int vector_id = blockIdx.x;

  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

  // reposition y to data for the current vector
  x += vector_id * V;

  typedef cub::BlockReduce<TopKMD<float, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  TopKMD<float, MAX_K> partial;
  bool finish = ALIVE ? false : finished[vector_id];
  for (int i = 0; i < MAX_K; ++i) {
    partial.topk.p[i] = -1;
    partial.topk.u[i] = -MAX_T_VAL;
  }
  partial.md.m = -MAX_T_VAL;
  partial.md.d = 0.0F;

  if (finish) {
    for (int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE) {
      float elem = (elem_id == E) ? MAX_T_VAL : -MAX_T_VAL;
      MD new_elem{elem, 1.0F};
      partial.md = reduce_md_op(partial.md, new_elem);
      partial.topk.insert(elem, elem_id);
    }
  } else {
    for (int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE) {
      float elem = (float)x[elem_id] + b[elem_id];
      MD new_elem{elem, 1.0F};
      partial.md = reduce_md_op(partial.md, new_elem);
      partial.topk.insert(elem, elem_id);
    }
  }

  TopKMD<float, MAX_K> total =
      BlockReduce(temp_storage)
          .Reduce(partial, reduce_topk_md_op<float, MAX_K>);

  if (thread_id == 0) {
    z += vector_id * K;
    v += vector_id * K;
    // c += vector_id;

    // cum_log_probs puts the results of alive beams after finish beams,
    // thus we add the offset.
    c += ALIVE ? (vector_id / (K / 2) * K + vector_id % (K / 2) + K / 2)
               : vector_id;

    // float d_total_inverse = __fdividef(1.0F, total.md.d);
    float d_total_log = logf(total.md.d);
    for (int i = 0; i < MAX_K; ++i) {
      // float val = __expf(total.topk.u[i] - total.md.m) * d_total_inverse;
      float val = total.topk.u[i] - total.md.m - d_total_log;
      if (i < K) {
        z[i] = total.topk.p[i] +
               vector_id * V;  // faster transformer needs absolute id
        v[i] = (float)val + (float)c[0];
      }
    }
  }
}

template <typename T,
          int ITEMS_PER_THREAD,
          int MAX_K,
          int THREADBLOCK_SIZE,
          bool ALIVE = false>
__launch_bounds__(THREADBLOCK_SIZE, 1) __global__
    void beam_online_softmax_topk_stage1_kernel(const T* __restrict x,
                                                const T* __restrict b,
                                                const bool* __restrict finished,
                                                float* __restrict t,
                                                int V,
                                                int K,
                                                int E) {
  int thread_id = threadIdx.x;
  int vector_id = blockIdx.x;

  const int PACKED_TOP_KMD_SIZE = 2 * MAX_K + 2;

  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

  // one will have multiple sections per V
  const int v_local = (V + gridDim.y - 1) / gridDim.y;
  const int section_start = v_local * blockIdx.y;
  int section_end = section_start + v_local;
  section_end = (section_end > V) ? V : section_end;

  // reposition x to data for the current vector
  x += vector_id * V;
#if TOPK_FP16_STORAGE == 1
  typedef cub::BlockReduce<TopKMD<__half, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
#else
  typedef cub::BlockReduce<TopKMD<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
#endif
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float buf_s[PACKED_TOP_KMD_SIZE];  // save intermediate result

#if TOPK_FP16_STORAGE == 1
  TopKMD<__half, MAX_K> partial;
#else
  TopKMD<T, MAX_K> partial;
#endif
  // bool finish = finished[vector_id];
  bool finish = ALIVE ? false : finished[vector_id];
  for (int i = 0; i < MAX_K; ++i) {
    partial.topk.p[i] = -1;
    partial.topk.u[i] = -MAX_T_VAL;
  }
  partial.md.m = -MAX_T_VAL;
  partial.md.d = 0.0F;

  if (finish) {
#pragma unroll 1
    for (int elem_id = section_start + thread_id; elem_id < section_end;
         elem_id += THREADBLOCK_SIZE) {
      float elem = (elem_id == E) ? MAX_T_VAL : -MAX_T_VAL;
      MD new_elem{elem, 1.0F};
      partial.md = reduce_md_op(partial.md, new_elem);
      partial.topk.insert(elem, elem_id);
    }
  } else {
#pragma unroll 1
    for (int elem_id = section_start + thread_id; elem_id < section_end;
         elem_id += THREADBLOCK_SIZE) {
      T bias = b == nullptr ? (T)0.0f : b[elem_id];  // gpt-2 does not use bias
      T elem = x[elem_id] + bias;
      MD new_elem{elem, 1.0F};
      partial.md = reduce_md_op(partial.md, new_elem);
      partial.topk.insert(elem, elem_id);
    }
  }

#if TOPK_FP16_STORAGE == 1
  TopKMD<__half, MAX_K> total =
      BlockReduce(temp_storage)
          .Reduce(partial, reduce_topk_md_op<__half, MAX_K>);
#else
  TopKMD<T, MAX_K> total =
      BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<T, MAX_K>);
#endif

  if (thread_id == 0) {
    for (int i = 0; i < K; i++) {
      reinterpret_cast<int*>(buf_s)[i] =
          total.topk.p[i] +
          vector_id * V;  // faster transformer needs absolute id
      buf_s[MAX_K + i] = total.topk.u[i];
    }
    buf_s[2 * MAX_K] = total.md.d;
    buf_s[2 * MAX_K + 1] = total.md.m;
  }
  __syncthreads();
  if (threadIdx.x < PACKED_TOP_KMD_SIZE) {
    t[blockIdx.x * PACKED_TOP_KMD_SIZE * gridDim.y +
      blockIdx.y * PACKED_TOP_KMD_SIZE + threadIdx.x] = buf_s[threadIdx.x];
  }
}

template <typename T, int MAX_K, int THREADBLOCK_SIZE, bool ALIVE = false>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void beam_online_softmax_topk_stage2_kernel(const float* __restrict x,
                                                const float* __restrict c,
                                                int* __restrict z,
                                                T* __restrict v,
                                                int K,
                                                int parts_per_beam) {
  const int vector_id = blockIdx.x;
  const int thread_id = threadIdx.x;
  const int PACKED_TOP_KMD_SIZE = 2 * MAX_K + 2;

  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

  extern __shared__ char buf_s_[];  // intermediate result
  float* buf_s = reinterpret_cast<float*>(buf_s_);
  //__shared__ float buf_s[PACKED_TOP_KMD_SIZE * THREADBLOCK_SIZE]; //
  // intermediate result

  typedef cub::BlockReduce<TopKMD<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  x += vector_id * PACKED_TOP_KMD_SIZE * parts_per_beam;

  TopKMD<T, MAX_K> partial;
  for (int i = 0; i < MAX_K; ++i) {
    partial.topk.p[i] = -1;
    partial.topk.u[i] = -MAX_T_VAL;
  }
  partial.md.m = -MAX_T_VAL;
  partial.md.d = 0.0F;

  // load and unpack into registers through smem
  for (int idx = thread_id; idx < PACKED_TOP_KMD_SIZE * parts_per_beam;
       idx += THREADBLOCK_SIZE) {
    buf_s[idx] = x[idx];
  }
  __syncthreads();

  if (threadIdx.x < parts_per_beam) {
    float* b_s = buf_s + thread_id * PACKED_TOP_KMD_SIZE;
    for (int i = 0; i < K; i++) {
      partial.topk.p[i] = reinterpret_cast<int*>(b_s)[i];
      partial.topk.u[i] = b_s[MAX_K + i];
    }
    partial.md.d = b_s[2 * MAX_K];
    partial.md.m = b_s[2 * MAX_K + 1];
  }
  __syncthreads();

  TopKMD<T, MAX_K> total =
      BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<T, MAX_K>);

  if (thread_id == 0) {
    z += vector_id * K;
    v += vector_id * K;
    // c += vector_id;

    // cum_log_probs puts the results of alive beams after finish beams,
    // thus we add the offset.
    c += ALIVE ? (vector_id / (K / 2) * K + vector_id % (K / 2) + K / 2)
               : vector_id;

    float d_total_log = logf(total.md.d);
    for (int i = 0; i < MAX_K; ++i) {
      float val = (float)total.topk.u[i] - total.md.m - d_total_log;
      if (i < K) {
        z[i] = total.topk.p[i];
        v[i] = (float)val + (float)c[0];
      }
    }
  }
}

template <typename T>
void topK_kernelLauncher(T* log_probs,
                         int* topk_tmp_id_buf,
                         T* topk_tmp_val_buf,
                         int* ids,
                         DecodingBeamsearchArguments args,
                         cudaStream_t stream) {
  const int batch_size = args.batch_size_;
  const int beam_width = args.beam_width_;
  const int vocab_size = args.vocab_size_padded_;
  const int diversity_rate = args.beam_search_diversity_rate_;
  const int block_size = SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE;

  switch (beam_width) {
    case 1:
      beam_topK_kernel<
          T,
          1,
          block_size><<<batch_size * beam_width, block_size, 0, stream>>>(
          log_probs,
          topk_tmp_id_buf,
          topk_tmp_val_buf,
          vocab_size,
          diversity_rate);
      batch_topK_kernel<T,
                        1,
                        block_size><<<batch_size, block_size, 0, stream>>>(
          topk_tmp_id_buf, topk_tmp_val_buf, ids);
      break;
    case 2:
      beam_topK_kernel<
          T,
          2,
          block_size><<<batch_size * beam_width, block_size, 0, stream>>>(
          log_probs,
          topk_tmp_id_buf,
          topk_tmp_val_buf,
          vocab_size,
          diversity_rate);
      batch_topK_kernel<T,
                        2,
                        block_size><<<batch_size, block_size, 0, stream>>>(
          topk_tmp_id_buf, topk_tmp_val_buf, ids);
      break;
    case 3:
      beam_topK_kernel<
          T,
          3,
          block_size><<<batch_size * beam_width, block_size, 0, stream>>>(
          log_probs,
          topk_tmp_id_buf,
          topk_tmp_val_buf,
          vocab_size,
          diversity_rate);
      batch_topK_kernel<T,
                        3,
                        block_size><<<batch_size, block_size, 0, stream>>>(
          topk_tmp_id_buf, topk_tmp_val_buf, ids);
      break;
    case 4:
      beam_topK_kernel<
          T,
          4,
          block_size><<<batch_size * beam_width, block_size, 0, stream>>>(
          log_probs,
          topk_tmp_id_buf,
          topk_tmp_val_buf,
          vocab_size,
          diversity_rate);
      batch_topK_kernel<T,
                        4,
                        block_size><<<batch_size, block_size, 0, stream>>>(
          topk_tmp_id_buf, topk_tmp_val_buf, ids);
      break;
    case 6:
      beam_topK_kernel<
          T,
          6,
          block_size><<<batch_size * beam_width, block_size, 0, stream>>>(
          log_probs,
          topk_tmp_id_buf,
          topk_tmp_val_buf,
          vocab_size,
          diversity_rate);
      batch_topK_kernel<T,
                        6,
                        block_size><<<batch_size, block_size, 0, stream>>>(
          topk_tmp_id_buf, topk_tmp_val_buf, ids);
      break;
    case 8:
      beam_topK_kernel<
          T,
          8,
          block_size><<<batch_size * beam_width, block_size, 0, stream>>>(
          log_probs,
          topk_tmp_id_buf,
          topk_tmp_val_buf,
          vocab_size,
          diversity_rate);
      batch_topK_kernel<T,
                        8,
                        block_size><<<batch_size, block_size, 0, stream>>>(
          topk_tmp_id_buf, topk_tmp_val_buf, ids);
      break;
    case 32:
      beam_topK_kernel<
          T,
          32,
          block_size><<<batch_size * beam_width, block_size, 0, stream>>>(
          log_probs,
          topk_tmp_id_buf,
          topk_tmp_val_buf,
          vocab_size,
          diversity_rate);
      batch_topK_kernel<T,
                        32,
                        block_size><<<batch_size, block_size, 0, stream>>>(
          topk_tmp_id_buf, topk_tmp_val_buf, ids);
      break;
    default:
      printf("[ERROR] Topk kernel does not support beamwidth = %d \n",
             beam_width);
      exit(0);
      break;
  }
}

template <typename T, int MAX_K, bool ALIVE = false>
void beam_online_softmax_topk_stage2_kernelLauncher(const float* temp_storage,
                                                    const float* cum_log_probs,
                                                    int* ids,
                                                    T* vals,
                                                    int batch_size,
                                                    int beam_width,
                                                    int parts_per_beam,
                                                    cudaStream_t stream) {
  // might rewrite beam_online_softmax_topk_stage2_kernel no to depend on
  // constant block size
  // in oreder to reduce compilation time
  int smem_stage2_size = parts_per_beam * (2 * MAX_K + 2) * sizeof(float);

  if (parts_per_beam <= 32) {
    beam_online_softmax_topk_stage2_kernel<
        T,
        MAX_K,
        32,
        ALIVE><<<batch_size * beam_width, 32, smem_stage2_size, stream>>>(
        temp_storage,
        cum_log_probs,
        ids,
        vals,
        ALIVE ? beam_width * 2 : beam_width,
        parts_per_beam);
    return;
  }
  if (parts_per_beam <= 64) {
    beam_online_softmax_topk_stage2_kernel<
        T,
        MAX_K,
        64,
        ALIVE><<<batch_size * beam_width, 64, smem_stage2_size, stream>>>(
        temp_storage,
        cum_log_probs,
        ids,
        vals,
        ALIVE ? beam_width * 2 : beam_width,
        parts_per_beam);
    return;
  }
  if (parts_per_beam <= 128) {
    beam_online_softmax_topk_stage2_kernel<
        T,
        MAX_K,
        128,
        ALIVE><<<batch_size * beam_width, 128, smem_stage2_size, stream>>>(
        temp_storage,
        cum_log_probs,
        ids,
        vals,
        ALIVE ? beam_width * 2 : beam_width,
        parts_per_beam);
    return;
  }
  assert(0);
}

template <typename T, int MAX_K>
void topK_softMax_kernelLauncher(const T* log_probs,
                                 const T* bias,
                                 const bool* finished,
                                 float* cum_log_probs,
                                 int* ids,
                                 void* temp_storage,
                                 const int temp_storage_size,
                                 const int batch_size,
                                 const int beam_width,
                                 const int vocab_size,
                                 const int end_id,
                                 T diversity_rate,
                                 cudaStream_t stream) {
  const int items_per_thread = 1;
  const int block_sz =
      (MAX_K < 16) ? (MAX_K < 8) ? SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE : 128
                   : 64;
  // const int block_sz = SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE;

  assert(temp_storage_size % 2 == 0);
  assert(temp_storage_size >= 2 * batch_size * beam_width * beam_width);

  const int topk_buf_offset =
      ceil(batch_size * beam_width * beam_width / 4.) * 4;
  int* topk_tmp_id_buf = reinterpret_cast<int*>(temp_storage);
  T* topk_tmp_val_buf = reinterpret_cast<T*>(topk_tmp_id_buf + topk_buf_offset);
  float* tmp_buffer =
      reinterpret_cast<float*>(topk_tmp_val_buf + topk_buf_offset);

#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
  int voc_parts = 4;
  if (batch_size * beam_width < 256) {
    // Volta has 80 SMs, so we aim for three waves
    voc_parts = (240 + batch_size * beam_width - 1) / (batch_size * beam_width);
    voc_parts = std::min(128, voc_parts);  // we implment up to 128
  }
  dim3 grid(batch_size * beam_width, voc_parts);
  cudaFuncSetAttribute(beam_online_softmax_topk_stage1_kernel<T,
                                                              items_per_thread,
                                                              MAX_K,
                                                              block_sz>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);
  beam_online_softmax_topk_stage1_kernel<
      T,
      items_per_thread,
      MAX_K,
      block_sz><<<grid, block_sz, 0, stream>>>(
      log_probs, bias, finished, tmp_buffer, vocab_size, beam_width, end_id);
#endif
  if (beam_width > 1) {
#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
    beam_online_softmax_topk_stage2_kernelLauncher<T, MAX_K>(tmp_buffer,
                                                             cum_log_probs,
                                                             topk_tmp_id_buf,
                                                             topk_tmp_val_buf,
                                                             batch_size,
                                                             beam_width,
                                                             voc_parts,
                                                             stream);
#else
    beam_online_softmax_topk_kernel<
        T,
        items_per_thread,
        MAX_K,
        block_sz><<<batch_size * beam_width, block_sz, 0, stream>>>(
        log_probs,
        bias,
        cum_log_probs,
        finished,
        topk_tmp_id_buf,
        topk_tmp_val_buf,
        vocab_size,
        beam_width,
        end_id);
#endif
#if 0
            // wrong result with diversity_rate != 0.f
            batch_topK_kernel<T, MAX_K, 32><<<batch_size, 32, 0, stream>>>
                                (topk_tmp_id_buf, topk_tmp_val_buf, ids, cum_log_probs);
#else
    batch_topk_kernel<T, MAX_K, 32><<<batch_size, 32, 0, stream>>>(
        topk_tmp_id_buf,
        topk_tmp_val_buf,
        ids,
        cum_log_probs,
        beam_width * beam_width,
        beam_width,
        diversity_rate);
#endif
  } else {
#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
    beam_online_softmax_topk_stage2_kernelLauncher<float, MAX_K>(tmp_buffer,
                                                                 cum_log_probs,
                                                                 ids,
                                                                 cum_log_probs,
                                                                 batch_size,
                                                                 beam_width,
                                                                 voc_parts,
                                                                 stream);
#else
    beam_online_softmax_topk_kernel<
        T,
        items_per_thread,
        MAX_K,
        block_sz><<<batch_size * beam_width, block_sz, 0, stream>>>(
        log_probs,
        bias,
        cum_log_probs,
        finished,
        ids,
        cum_log_probs,
        vocab_size,
        beam_width,
        end_id);
#endif
  }
}

template <typename T>
void topK_softMax(const T* log_probs,
                  const T* bias,
                  const bool* finished,
                  float* cum_log_probs,
                  int* ids,
                  void* temp_storage,
                  DecodingBeamsearchArguments args,
                  cudaStream_t stream) {
  const int temp_storage_size = args.temp_storage_size_;
  const int batch_size = args.batch_size_;
  const int beam_width = args.beam_width_;
  const int vocab_size = args.vocab_size_padded_;
  const int end_id = args.end_id_;
  const T diversity_rate = args.beam_search_diversity_rate_;

  switch (beam_width) {
    case 1:
      topK_softMax_kernelLauncher<T, 1>(log_probs,
                                        bias,
                                        finished,
                                        cum_log_probs,
                                        ids,
                                        temp_storage,
                                        temp_storage_size,
                                        batch_size,
                                        beam_width,
                                        vocab_size,
                                        end_id,
                                        diversity_rate,
                                        stream);
      break;
    case 2:
      topK_softMax_kernelLauncher<T, 2>(log_probs,
                                        bias,
                                        finished,
                                        cum_log_probs,
                                        ids,
                                        temp_storage,
                                        temp_storage_size,
                                        batch_size,
                                        beam_width,
                                        vocab_size,
                                        end_id,
                                        diversity_rate,
                                        stream);
      break;
    case 3:
      topK_softMax_kernelLauncher<T, 3>(log_probs,
                                        bias,
                                        finished,
                                        cum_log_probs,
                                        ids,
                                        temp_storage,
                                        temp_storage_size,
                                        batch_size,
                                        beam_width,
                                        vocab_size,
                                        end_id,
                                        diversity_rate,
                                        stream);
      break;
    case 4:
      topK_softMax_kernelLauncher<T, 4>(log_probs,
                                        bias,
                                        finished,
                                        cum_log_probs,
                                        ids,
                                        temp_storage,
                                        temp_storage_size,
                                        batch_size,
                                        beam_width,
                                        vocab_size,
                                        end_id,
                                        diversity_rate,
                                        stream);
      break;
    case 8:
      topK_softMax_kernelLauncher<T, 8>(log_probs,
                                        bias,
                                        finished,
                                        cum_log_probs,
                                        ids,
                                        temp_storage,
                                        temp_storage_size,
                                        batch_size,
                                        beam_width,
                                        vocab_size,
                                        end_id,
                                        diversity_rate,
                                        stream);
      break;
    case 16:
      topK_softMax_kernelLauncher<T, 16>(log_probs,
                                         bias,
                                         finished,
                                         cum_log_probs,
                                         ids,
                                         temp_storage,
                                         temp_storage_size,
                                         batch_size,
                                         beam_width,
                                         vocab_size,
                                         end_id,
                                         diversity_rate,
                                         stream);
      break;
    case 32:
      topK_softMax_kernelLauncher<T, 32>(log_probs,
                                         bias,
                                         finished,
                                         cum_log_probs,
                                         ids,
                                         temp_storage,
                                         temp_storage_size,
                                         batch_size,
                                         beam_width,
                                         vocab_size,
                                         end_id,
                                         diversity_rate,
                                         stream);
      break;
    default:
      printf("[ERROR] Topk kernel does not support beamwidth = %d \n",
             beam_width);
      exit(0);
      break;
  }
}

template void topK_kernelLauncher<float>(float* log_probs,
                                         int* topk_tmp_id_buf,
                                         float* topk_tmp_val_buf,
                                         int* ids,
                                         DecodingBeamsearchArguments args,
                                         cudaStream_t stream);

template void topK_kernelLauncher<half>(half* log_probs,
                                        int* topk_tmp_id_buf,
                                        half* topk_tmp_val_buf,
                                        int* ids,
                                        DecodingBeamsearchArguments args,
                                        cudaStream_t stream);

template void topK_softMax<float>(const float* log_probs,
                                  const float* bias,
                                  const bool* finished,
                                  float* cum_log_probs,
                                  int* ids,
                                  void* tmp_storage,
                                  DecodingBeamsearchArguments args,
                                  cudaStream_t stream);

template void topK_softMax<half>(const half* log_probs,
                                 const half* bias,
                                 const bool* finished,
                                 float* cum_log_probs,
                                 int* ids,
                                 void* tmp_storage,
                                 DecodingBeamsearchArguments args,
                                 cudaStream_t stream);

template <typename T, int MAX_K>
struct TopKFinish {
  T u[MAX_K];
  int idx[MAX_K];
  int len[MAX_K];

  __device__ __forceinline__ void insert(T elem, int pidx, int step) {
    if (elem > u[MAX_K - 1]) {
      u[MAX_K - 1] = elem;
      idx[MAX_K - 1] = pidx;
      len[MAX_K - 1] = step;

      for (int k = MAX_K - 2; k >= 0; --k) {
        if (u[k + 1] > u[k]) {
          T u2 = u[k];
          u[k] = u[k + 1];
          u[k + 1] = u2;
          int tmp = idx[k];
          idx[k] = idx[k + 1];
          idx[k + 1] = tmp;
          tmp = len[k];
          len[k] = len[k + 1];
          len[k + 1] = tmp;
        }
      }
    }
  }
};

template <int MAX_K>
struct TopKStop {
  float u[MAX_K];
  int word_id[MAX_K];
  int idx[MAX_K];
  int len[MAX_K];

  __device__ __forceinline__ void insert(float elem,
                                         int ids,
                                         int pidx,
                                         int step) {
    if (elem > u[MAX_K - 1]) {
      u[MAX_K - 1] = elem;
      word_id[MAX_K - 1] = ids;
      idx[MAX_K - 1] = pidx;
      len[MAX_K - 1] = step;

      for (int k = MAX_K - 2; k >= 0; --k) {
        if (u[k + 1] > u[k]) {
          float u2 = u[k];
          u[k] = u[k + 1];
          u[k + 1] = u2;

          int tmp1 = word_id[k];
          word_id[k] = word_id[k + 1];
          word_id[k + 1] = tmp1;

          int tmp2 = idx[k];
          idx[k] = idx[k + 1];
          idx[k + 1] = tmp2;

          int tmp3 = len[k];
          len[k] = len[k + 1];
          len[k + 1] = tmp3;
        }
      }
    }
  }
};

template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void batch_topk_update_kernel(const int* __restrict x,
                                  const T* __restrict y,
                                  bool* finished,
                                  bool* alive_finished,
                                  int* sequence_length,
                                  int* word_ids,
                                  int* parent_ids,
                                  int* output_word_ids,
                                  int* output_parent_ids,
                                  float* output_cum_log_probs,
                                  const int batch_size,
                                  const int beam_width,
                                  const int vocab_size,
                                  const int end_id,
                                  const int step,
                                  const int max_out_len,
                                  int V,
                                  int K,
                                  T diversity_rate,
                                  float length_penalty,
                                  float max_length_penalty,
                                  const int finished_candidate_num,
                                  const bool early_stopping = false) {
  int thread_id = threadIdx.x;
  int vector_id = blockIdx.x;

  const bool IS_FP16 = std::is_same<T, half>::value;
  // to be consistent with MAX_T_VAL in init_kernel, which should also be same
  // with other topk kernel, however it does not
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : 1e20f;

  // reposition x, y to data for the current vector
  x += vector_id * V;
  y += vector_id * V;

  typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  TopK<T, MAX_K> partial;
  for (int i = 0; i < MAX_K; ++i) {
    partial.p[i] = -1;
    partial.u[i] = -MAX_T_VAL;
  }
  for (int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE) {
    int i = elem_id % K;
    T elem = y[elem_id] + diversity_rate * (T)i;
    int elem_idx = elem_id;  // x[elem_id];
    partial.insert(elem, elem_idx);
  }

  TopK<T, MAX_K> total =
      BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

  // grow finish and grow alive
  if (thread_id == 0) {
    word_ids += vector_id * beam_width;
    parent_ids += vector_id * beam_width;
    output_word_ids += vector_id * K;  // K == MAX_K == beam_width*2
    output_parent_ids += vector_id * K;
    output_cum_log_probs += vector_id * K;
    sequence_length += vector_id * K;
    finished += vector_id * K;
    alive_finished += vector_id * beam_width;
    // load the finish queue to grow
    // TODO(guosheng): Use vectorized load or do this BlockReduce to use
    // multi-threads without extra sync
    int finish_num = 0;
    TopKFinish<T, MAX_K / 2> finish_candidate;
    for (int i = 0; i < MAX_K / 2; ++i) {
      if (step == 1) {  // step number starts from 1 rather than 0
        finish_candidate.u[i] = -MAX_T_VAL;
        finish_candidate.idx[i] = -1;
        finish_candidate.len[i] = 0;
      } else {
        finish_candidate.u[i] = output_cum_log_probs[i];
        finish_candidate.idx[i] = i;
        finish_candidate.len[i] = sequence_length[i];
        if (finished[i]) finish_num++;
      }
    }

    int alive_num = 0;
    for (int i = 0; i < MAX_K; ++i) {  // K == MAX_K == beam_width*2
      if (i < K) {
        // beam_online_softmax_topk_kernel produces absolute id, which can make
        // update_KV_cache_kernel use gather instead of gather_nd
        int abs_id = x[total.p[i]];
        float cum_log_prob = (float)y[total.p[i]];
        // There are two queues, one for the alive and another for the finish.
        // `beam_id` stands for parents in the alive, and it uses absolute id
        // represented as `batch_idx * beam_width + beam_idx`.
        int beam_id = abs_id / vocab_size;
        int beam_id_in_output =
            vector_id * K + (beam_id % beam_width) + beam_width;
        int word_id = abs_id % vocab_size;
        if (i < finished_candidate_num && word_id == end_id) {  // grow finish
          // The alive candidates are put after finish candidates in the
          // finish queue, thus parent index should plus with the
          // offset(beam_width).
          finish_candidate.insert(
              cum_log_prob / length_penalty, beam_id_in_output, step);
          if (finish_num != MAX_K / 2) finish_num++;
        } else if (alive_num < beam_width && word_id != end_id) {  // grow alive
          parent_ids[alive_num] = beam_id;
          word_ids[alive_num] = word_id;
          // Also put alive candidates after finish candidates, since output
          // must include both the finish and alive to trace full path
          output_word_ids[MAX_K / 2 + alive_num] = word_id;
          output_parent_ids[MAX_K / 2 + alive_num] = beam_id_in_output;
          output_cum_log_probs[MAX_K / 2 + alive_num] = cum_log_prob;
          sequence_length[MAX_K / 2 + alive_num] = step;
          finished[MAX_K / 2 + alive_num] = 0;
          alive_finished[alive_num] = 0;
          alive_num++;
        }
      }
    }

    for (int i = 0; i < MAX_K / 2; ++i) {
      output_word_ids[i] = end_id;
      output_cum_log_probs[i] = static_cast<float>(finish_candidate.u[i]);
      output_parent_ids[i] = finish_candidate.idx[i];
      sequence_length[i] = finish_candidate.len[i];
      // finished[i] = 1;
      finished[i] = finish_candidate.u[i] > (-MAX_T_VAL + (T)10.0f) ? 1 : 0;
    }

    // early finish
    float lowest_finish =
        finish_num == 0 ? -1e20f : output_cum_log_probs[finish_num - 1];
    // The best possible score of the most likely alive sequence
    float lower_bound =
        (float)output_cum_log_probs[MAX_K / 2] / max_length_penalty;

    // output must include both the finish and alive to trace full path
    if (finished_candidate_num == MAX_K / 2) {
      if (finish_num == finished_candidate_num &&
          (lowest_finish > lower_bound || early_stopping)) {  // when finishing
        // If early stop, also mark the alive beams finished.
        for (int i = MAX_K / 2; i < MAX_K; ++i) {
          finished[i] = 1;
          alive_finished[i - MAX_K / 2] = 1;
        }
      } else if (step == max_out_len) {
        TopKStop<MAX_K / 2> finish_stop;
        for (int i = 0; i < MAX_K / 2; ++i) {
          finish_stop.word_id[i] = -1;
          finish_stop.u[i] = -1e20f;
          finish_stop.idx[i] = -1;
          finish_stop.len[i] = 0;
        }

        for (int i = 0; i < finish_num; ++i) {
          finish_stop.insert(output_cum_log_probs[i],
                             end_id,
                             output_parent_ids[i],
                             sequence_length[i]);
        }
        for (int i = MAX_K / 2; i < MAX_K; ++i) {
          finish_stop.insert(output_cum_log_probs[i] / length_penalty,
                             word_ids[i - MAX_K / 2],
                             output_parent_ids[i],
                             step);
        }
        for (int i = 0; i < MAX_K / 2; ++i) {
          output_word_ids[i] = finish_stop.word_id[i];
          output_cum_log_probs[i] = finish_stop.u[i];
          output_parent_ids[i] = finish_stop.idx[i];
          sequence_length[i] = finish_stop.len[i];
          finished[i] = 1;
        }

        // If early stop, also mark the alive beams finished.
        for (int i = MAX_K / 2; i < MAX_K; ++i) {
          finished[i] = 1;
          alive_finished[i - MAX_K / 2] = 1;
        }
      }
    } else {
      if (step == max_out_len ||
          lowest_finish > lower_bound) {  // when finishing
        for (int i = 0; finish_num < MAX_K / 2; ++finish_num, ++i) {
          output_word_ids[finish_num] = word_ids[i];
          output_cum_log_probs[finish_num] =
              (float)output_cum_log_probs[i + beam_width] / length_penalty;
          output_parent_ids[finish_num] = output_parent_ids[i + beam_width];
          sequence_length[finish_num] = step;
          finished[finish_num] = 1;
        }
        // If early stop, also mark the alive beams finished.
        for (int i = MAX_K / 2; i < MAX_K; ++i) {
          finished[i] = 1;
          alive_finished[i - MAX_K / 2] = 1;
        }
      }
    }
  }
}

template <typename T, int MAX_K>
void topK_softMax_update_kernelLauncher(const T* log_probs,
                                        const T* bias,
                                        bool* finished,
                                        bool* alive_finished,
                                        int* sequence_length,
                                        int* word_ids,
                                        int* parent_ids,
                                        int* output_word_ids,
                                        int* output_parent_ids,
                                        float* cum_log_probs,
                                        void* temp_storage,
                                        const int temp_storage_size,
                                        const int batch_size,
                                        const int beam_width,
                                        const int vocab_size,
                                        const int end_id,
                                        const int step,
                                        const int max_out_len,
                                        T diversity_rate,
                                        const float alpha,
                                        const int finished_candidate_num,
                                        const bool early_stopping,
                                        cudaStream_t stream) {
  const int items_per_thread = 1;
  const int block_sz =
      (MAX_K < 16) ? (MAX_K < 8) ? SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE : 128
                   : 64;
  // const int block_sz = SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE;

  assert(temp_storage_size % 2 == 0);
  // select top beam_width*2 for topk_tmp_id_buf and topk_tmp_val_buf
  assert(temp_storage_size >= 2 * batch_size * beam_width * beam_width * 2);

  const int topk_buf_offset =
      ceil(batch_size * beam_width * beam_width / 4.) * 4 * 2;
  int* topk_tmp_id_buf = reinterpret_cast<int*>(temp_storage);
  T* topk_tmp_val_buf = reinterpret_cast<T*>(topk_tmp_id_buf + topk_buf_offset);
  float* tmp_buffer =
      reinterpret_cast<float*>(topk_tmp_val_buf + topk_buf_offset);

#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
  int voc_parts = 4;
  if (batch_size * beam_width < 256) {
    // Volta has 80 SMs, so we aim for three waves
    voc_parts = (240 + batch_size * beam_width - 1) / (batch_size * beam_width);
    voc_parts = std::min(128, voc_parts);  // we implment up to 128
  }
  dim3 grid(batch_size * beam_width, voc_parts);
  cudaFuncSetAttribute(beam_online_softmax_topk_stage1_kernel<T,
                                                              items_per_thread,
                                                              MAX_K,
                                                              block_sz>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);
  beam_online_softmax_topk_stage1_kernel<T,
                                         items_per_thread,
                                         MAX_K,
                                         block_sz,
                                         true><<<grid, block_sz, 0, stream>>>(
      log_probs,
      bias,
      finished,
      tmp_buffer,
      vocab_size,
      beam_width * 2,
      end_id);
  beam_online_softmax_topk_stage2_kernelLauncher<T, MAX_K, true>(
      tmp_buffer,
      cum_log_probs,
      topk_tmp_id_buf,
      topk_tmp_val_buf,
      batch_size,
      beam_width,
      voc_parts,
      stream);  // double beam_width in launcher
#else
  beam_online_softmax_topk_kernel<
      T,
      items_per_thread,
      MAX_K,
      block_sz,
      true><<<batch_size * beam_width, block_sz, 0, stream>>>(log_probs,
                                                              bias,
                                                              cum_log_probs,
                                                              finished,
                                                              topk_tmp_id_buf,
                                                              topk_tmp_val_buf,
                                                              vocab_size,
                                                              beam_width * 2,
                                                              end_id);
#endif
  float length_penalty = (finished_candidate_num == beam_width)
                             ? std::pow((5. + step - 1) / 6., alpha)
                             : std::pow((5. + step + 1) / 6., alpha);
  float max_length_penalty = (finished_candidate_num == beam_width)
                                 ? length_penalty
                                 : std::pow((5. + max_out_len + 1) / 6., alpha);
  batch_topk_update_kernel<T, MAX_K, 32><<<batch_size, 32, 0, stream>>>(
      topk_tmp_id_buf,
      topk_tmp_val_buf,
      finished,
      alive_finished,
      sequence_length,
      word_ids,
      parent_ids,
      output_word_ids,
      output_parent_ids,
      cum_log_probs,
      batch_size,
      beam_width,
      vocab_size,
      end_id,
      step,
      max_out_len,
      beam_width * beam_width * 2,
      beam_width * 2,
      diversity_rate,
      length_penalty,
      max_length_penalty,
      finished_candidate_num,
      early_stopping);
}

template <typename T>
void topK_softMax_update(
    const T* log_probs,
    const T* bias,  // NOTE: bias is float in V3.1
    bool* finished,
    bool* alive_finished,
    int* sequence_length,
    int* word_ids,
    int* parent_ids,  // for update cache, only include alive beams
    int* output_word_ids,
    int* output_parent_ids,  // for gather tree, include both alive and finish
                             // beams
    float* output_cum_log_probs,  // NOTE: cum_log_probs is T in V3.1
    void* temp_storage,
    const int step,
    DecodingBeamsearchArguments args,
    cudaStream_t stream) {
  const int temp_storage_size = args.temp_storage_size_;
  const int batch_size = args.batch_size_;
  const int beam_width = args.beam_width_;
  const int vocab_size = args.vocab_size_padded_;
  const int end_id = args.end_id_;
  const T diversity_rate = args.beam_search_diversity_rate_;
  const int max_out_len = args.seq_len_;
  const float alpha = args.alpha_;
  const int finished_candidate_num = args.finished_candidate_num_;
  const bool early_stopping = args.early_stopping_;

  switch (beam_width) {
    case 1:
      topK_softMax_update_kernelLauncher<T, 2>(log_probs,
                                               bias,
                                               finished,
                                               alive_finished,
                                               sequence_length,
                                               word_ids,
                                               parent_ids,
                                               output_word_ids,
                                               output_parent_ids,
                                               output_cum_log_probs,
                                               temp_storage,
                                               temp_storage_size,
                                               batch_size,
                                               beam_width,
                                               vocab_size,
                                               end_id,
                                               step,
                                               max_out_len,
                                               diversity_rate,
                                               alpha,
                                               finished_candidate_num,
                                               early_stopping,
                                               stream);
      break;
    case 2:
      topK_softMax_update_kernelLauncher<T, 4>(log_probs,
                                               bias,
                                               finished,
                                               alive_finished,
                                               sequence_length,
                                               word_ids,
                                               parent_ids,
                                               output_word_ids,
                                               output_parent_ids,
                                               output_cum_log_probs,
                                               temp_storage,
                                               temp_storage_size,
                                               batch_size,
                                               beam_width,
                                               vocab_size,
                                               end_id,
                                               step,
                                               max_out_len,
                                               diversity_rate,
                                               alpha,
                                               finished_candidate_num,
                                               early_stopping,
                                               stream);
      break;
    case 3:
      topK_softMax_update_kernelLauncher<T, 6>(log_probs,
                                               bias,
                                               finished,
                                               alive_finished,
                                               sequence_length,
                                               word_ids,
                                               parent_ids,
                                               output_word_ids,
                                               output_parent_ids,
                                               output_cum_log_probs,
                                               temp_storage,
                                               temp_storage_size,
                                               batch_size,
                                               beam_width,
                                               vocab_size,
                                               end_id,
                                               step,
                                               max_out_len,
                                               diversity_rate,
                                               alpha,
                                               finished_candidate_num,
                                               early_stopping,
                                               stream);
      break;
    case 4:
      topK_softMax_update_kernelLauncher<T, 8>(log_probs,
                                               bias,
                                               finished,
                                               alive_finished,
                                               sequence_length,
                                               word_ids,
                                               parent_ids,
                                               output_word_ids,
                                               output_parent_ids,
                                               output_cum_log_probs,
                                               temp_storage,
                                               temp_storage_size,
                                               batch_size,
                                               beam_width,
                                               vocab_size,
                                               end_id,
                                               step,
                                               max_out_len,
                                               diversity_rate,
                                               alpha,
                                               finished_candidate_num,
                                               early_stopping,
                                               stream);
      break;
    case 8:
      topK_softMax_update_kernelLauncher<T, 16>(log_probs,
                                                bias,
                                                finished,
                                                alive_finished,
                                                sequence_length,
                                                word_ids,
                                                parent_ids,
                                                output_word_ids,
                                                output_parent_ids,
                                                output_cum_log_probs,
                                                temp_storage,
                                                temp_storage_size,
                                                batch_size,
                                                beam_width,
                                                vocab_size,
                                                end_id,
                                                step,
                                                max_out_len,
                                                diversity_rate,
                                                alpha,
                                                finished_candidate_num,
                                                early_stopping,
                                                stream);
      break;
    case 16:
      topK_softMax_update_kernelLauncher<T, 32>(log_probs,
                                                bias,
                                                finished,
                                                alive_finished,
                                                sequence_length,
                                                word_ids,
                                                parent_ids,
                                                output_word_ids,
                                                output_parent_ids,
                                                output_cum_log_probs,
                                                temp_storage,
                                                temp_storage_size,
                                                batch_size,
                                                beam_width,
                                                vocab_size,
                                                end_id,
                                                step,
                                                max_out_len,
                                                diversity_rate,
                                                alpha,
                                                finished_candidate_num,
                                                early_stopping,
                                                stream);
      break;
    case 32:
      topK_softMax_update_kernelLauncher<T, 64>(log_probs,
                                                bias,
                                                finished,
                                                alive_finished,
                                                sequence_length,
                                                word_ids,
                                                parent_ids,
                                                output_word_ids,
                                                output_parent_ids,
                                                output_cum_log_probs,
                                                temp_storage,
                                                temp_storage_size,
                                                batch_size,
                                                beam_width,
                                                vocab_size,
                                                end_id,
                                                step,
                                                max_out_len,
                                                diversity_rate,
                                                alpha,
                                                finished_candidate_num,
                                                early_stopping,
                                                stream);
      break;
    default:
      printf("[ERROR] Topk kernel does not support beamwidth = %d \n",
             beam_width);
      exit(0);
      break;
  }
}

template void topK_softMax_update<float>(
    const float* log_probs,
    const float* bias,
    bool* finished,
    bool* alive_finished,
    int* sequence_length,
    int* word_ids,
    int* parent_ids,  // for update cache, only include alive beams
    int* output_word_ids,
    int* output_parent_ids,  // for gather tree, include both alive and finish
                             // beams
    float* output_cum_log_probs,
    void* temp_storage,
    const int step,
    DecodingBeamsearchArguments args,
    cudaStream_t stream);

template void topK_softMax_update<half>(
    const half* log_probs,
    const half* bias,
    bool* finished,
    bool* alive_finished,
    int* sequence_length,
    int* word_ids,
    int* parent_ids,  // for update cache, only include alive beams
    int* output_word_ids,
    int* output_parent_ids,  // for gather tree, include both alive and finish
                             // beams
    float* output_cum_log_probs,
    void* temp_storage,
    const int step,
    DecodingBeamsearchArguments args,
    cudaStream_t stream);

}  // end of namespace fastertransformer
