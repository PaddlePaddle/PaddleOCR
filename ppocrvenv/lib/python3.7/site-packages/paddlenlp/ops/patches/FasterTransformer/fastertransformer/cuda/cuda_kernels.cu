/*
 * Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

template <typename T, bool ALIVE>
__global__ void update_logits_kernel(T* logits,
                                     const T* bias,
                                     const int end_id,
                                     const bool* finished,
                                     const int n) {
  int bid = blockIdx.x;
  bool finish = ALIVE ? false : finished[bid];
  int offset = bid * n;

  const T MAX_T_VAL = (sizeof(T) == 2) ? HALF_FLT_MAX : FLT_MAX;

  float max_val = -FLT_MAX;
  __shared__ float s_max_val;
  __shared__ float s_sum_val;

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    if (finish)
      logits[offset + tid] = (tid == end_id) ? MAX_T_VAL : -MAX_T_VAL;
    else
      logits[offset + tid] += bias[tid];
    max_val = max(max_val, (float)logits[offset + tid]);
  }

  max_val = blockReduceMax<float>((float)max_val);
  if (threadIdx.x == 0) s_max_val = max_val;
  __syncthreads();

  float sum_val = 0.0f;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    float tmp = __expf((float)logits[offset + tid] - s_max_val);
    logits[offset + tid] = (T)tmp;
    sum_val += tmp;
  }

  sum_val = blockReduceSum<float>(sum_val);
  if (threadIdx.x == 0) s_sum_val = sum_val;
  __syncthreads();

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    logits[offset + tid] = (T)logf((float)logits[offset + tid] / s_sum_val);
  }
}

template <typename T>
void update_logits_v2(T* logits,
                      const T* bias,
                      const int end_id,
                      const bool* finished,
                      const int m,
                      const int n,
                      cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));
  /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big.
   */
  update_logits_kernel<T, true><<<grid, block, 0, stream>>>(
      logits, bias, end_id, finished, n);
}

template void update_logits_v2(float* logits,
                               const float* bias,
                               const int end_id,
                               const bool* finished,
                               const int m,
                               const int n,
                               cudaStream_t stream);

template void update_logits_v2(half* logits,
                               const half* bias,
                               const int end_id,
                               const bool* finished,
                               const int m,
                               const int n,
                               cudaStream_t stream);
}  // namespace fastertransformer
