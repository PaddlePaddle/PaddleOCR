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

#include "fastertransformer/cuda/transformer_kernels.cuh"

namespace fastertransformer {


template <typename T>
__inline__ __device__ T gelu(T x) {
  float cdf =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));

  // NOTE: The precision of gelu with or without approximate formulation
  // may cause serious problem in some cases. If necessary, the following
  // comments can be opened to use the non-approximate formulation.
  // float cdf = 0.5f * (1.0f + erf((float)x / sqrt(2.0f)));
  return x * cdf;
}

template <>
__inline__ __device__ half2 gelu(half2 val) {
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp = __half22float2(val);

  tmp.x =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));
}

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}

template <typename T>
__global__ void add_bias_gelu(T* out, const T* __restrict bias, int m, int n) {
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n;
       id += blockDim.x * gridDim.x) {
    T reg_bias = __ldg(&bias[id % n]);
    T val = out[id] + reg_bias;
    out[id] = (T)(gelu(val));
  }
}

template <>
__global__ void add_bias_gelu(half* out,
                              const half* __restrict bias,
                              int m,
                              int n) {
  half2* out_ptr = (half2*)out;
  const half2* bias_ptr = (half2*)bias;

  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n;
       id += blockDim.x * gridDim.x) {
    half2 reg_bias = __ldg(&bias_ptr[id % n]);
    half2 val = out_ptr[id] + reg_bias;
    out_ptr[id] = gelu(val);
  }
}

template <typename T>
__global__ void add_bias_relu(T* out, const T* __restrict bias, int m, int n) {
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n;
       id += blockDim.x * gridDim.x) {
    T reg_bias = __ldg(&bias[id % n]);
    T val = out[id] + reg_bias;
    out[id] = (T)(val > 0.0f ? val : 0.0f);
  }
}

template <>
__global__ void add_bias_relu(half* out,
                              const half* __restrict bias,
                              int m,
                              int n) {
  half2* out_ptr = (half2*)out;
  const half2* bias_ptr = (half2*)bias;

  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n;
       id += blockDim.x * gridDim.x) {
    half2 reg_bias = __ldg(&bias_ptr[id % n]);
    half2 val = out_ptr[id] + reg_bias;
    val.x = val.x > (half)0.0f ? val.x : (half)0.0f;
    val.y = val.y > (half)0.0f ? val.y : (half)0.0f;
    out_ptr[id] = val;
  }
}

template <typename T>
__global__ void add_bias_input_layernorm(T* out,
                                         const T* input,
                                         const T* bias,
                                         const T* gamma,
                                         const T* beta,
                                         int m,
                                         int n) {
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  local_out += (float)(out[blockIdx.x * n + tid] + input[blockIdx.x * n + tid] +
                       __ldg(&bias[tid]));

  mean = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0) s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
  if (threadIdx.x == 0) s_variance = variance / n + 1e-6f;
  __syncthreads();

  out[blockIdx.x * n + tid] = (T)(((local_out - s_mean) * rsqrtf(s_variance)) *
                                      (float)(__ldg(&gamma[tid])) +
                                  (float)(__ldg(&beta[tid])));
}

template <>
__global__ void add_bias_input_layernorm(half* out,
                                         const half* input,
                                         const half* bias,
                                         const half* gamma,
                                         const half* beta,
                                         int m,
                                         int n) {
  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;
  float2 local_out_fp2;

  half2* out_ptr = (half2*)out;
  const half2* input_ptr = (const half2*)input;
  const half2* bias_ptr = (const half2*)bias;
  const half2* gamma_ptr = (const half2*)gamma;
  const half2* beta_ptr = (const half2*)beta;

  float local_out = 0.0f;
  int id = blockIdx.x * n / 2 + tid;
  local_out_fp2 = __half22float2(
      __hadd2(__hadd2(out_ptr[id], input_ptr[id]), __ldg(&bias_ptr[tid])));
  local_out += local_out_fp2.x;
  local_out += local_out_fp2.y;

  mean = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0) s_mean = mean / n;
  __syncthreads();

  variance = (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
  variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
  float2 beta_val = __half22float2(__ldg(&beta_ptr[tid]));
  local_out_fp2.x =
      (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
  local_out_fp2.y =
      (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
  out_ptr[id] = __float22half2_rn(local_out_fp2);
}


template <typename T>
__global__ void add_bias_input_layernorm_v2(T* out,
                                            const T* __restrict input,
                                            const T* __restrict bias,
                                            const T* __restrict gamma,
                                            const T* __restrict beta,
                                            int n) {
  const int ite = 4;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;
  float local_out[ite];

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = bid * n + col_id;
    local_out[i] = (float)(out[id] + __ldg(&input[id]) + __ldg(&bias[col_id]));
    sum += local_out[i];
  }

  mean = blockReduceSum<float>(sum);
  if (tid == 0) s_mean = mean / n;
  __syncthreads();

  float var = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    float diff = local_out[i] - s_mean;
    var += diff * diff;
  }

  variance = blockReduceSum<float>(var);
  if (tid == 0) s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = bid * n + col_id;
    out[id] = (T)((local_out[i] - s_mean) * s_variance *
                      (float)__ldg(&gamma[col_id]) +
                  (float)__ldg(&beta[col_id]));
  }
}

template <>
__global__ void add_bias_input_layernorm_v2(half* out,
                                            const half* __restrict input,
                                            const half* __restrict bias,
                                            const half* __restrict gamma,
                                            const half* __restrict beta,
                                            int n) {
  const int ite = 4;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;
  half2 local_out_half2[ite];

  half2* out_ptr = (half2*)out;
  const half2* input_ptr = (const half2*)input;
  const half2* bias_ptr = (const half2*)bias;
  const half2* gamma_ptr = (const half2*)gamma;
  const half2* beta_ptr = (const half2*)beta;

  // float sum = 0.0f;
  half2 sum = __float2half2_rn(0.0f);
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = bid * n / 2 + col_id;
    local_out_half2[i] =
        out_ptr[id] + __ldg(&input_ptr[id]) + __ldg(&bias_ptr[col_id]);
    sum += local_out_half2[i];
  }

  mean = blockReduceSum<float>((float)(sum.x + sum.y));
  if (threadIdx.x == 0) s_mean = mean / n;
  __syncthreads();

  float var = 0.0f;
  half2 s_mean_2 = __float2half2_rn(s_mean);
#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out_half2[i] = local_out_half2[i] - s_mean_2;
    float v1 = (float)local_out_half2[i].x;
    float v2 = (float)local_out_half2[i].y;
    var += v1 * v1 + v2 * v2;
  }

  variance = blockReduceSum<float>(var);
  if (threadIdx.x == 0) s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  half2 s_var_2 = __float2half2_rn(s_variance);
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + tid;
    int id = bid * n / 2 + col_id;
    out_ptr[id] = local_out_half2[i] * s_var_2 * __ldg(&gamma_ptr[col_id]) +
                  __ldg(&beta_ptr[col_id]);
  }
}

template <typename T>
void add_bias_act_kernelLauncher(T* out,
                                 const T* bias,
                                 int m,
                                 int n,
                                 ActivationType activation_type,
                                 cudaStream_t stream) {
  const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16
  dim3 block, grid;
  if (n / 4 / data_type_factor <= 1024) {
    block.x = n / 4 / data_type_factor;
    grid.x = m;
  } else {
    block.x = 1024;
    grid.x = ceil(m * n / 1024.);
  }


  if (activation_type == ActivationType::RELU)
    add_bias_relu<T><<<grid, block, 0, stream>>>(
        out, bias, m, n / data_type_factor);
  else if (activation_type == ActivationType::GELU)
    add_bias_gelu<T><<<grid, block, 0, stream>>>(
        out, bias, m, n / data_type_factor);
}

template <typename T>
void add_bias_input_layernorm_kernelLauncher(T* out,
                                             const T* input,
                                             const T* bias,
                                             const T* gamma,
                                             const T* beta,
                                             int m,
                                             int n,
                                             cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  if (n == 768 || n == 1024)
    add_bias_input_layernorm_v2<T><<<grid, n / 4, 0, stream>>>(
        out, input, bias, gamma, beta, n);
  else
    add_bias_input_layernorm<T><<<grid, block, 0, stream>>>(
        out, input, bias, gamma, beta, m, n);
}

template <>
void add_bias_input_layernorm_kernelLauncher(half* out,
                                             const half* input,
                                             const half* bias,
                                             const half* gamma,
                                             const half* beta,
                                             int m,
                                             int n,
                                             cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(n / 2);
  assert(n / 2 <= 1024);

  if (m >= 512 && (n == 768 || n == 1024))
    add_bias_input_layernorm_v2<half><<<grid, n / 8, 0, stream>>>(
        out, input, bias, gamma, beta, n);
  else
    add_bias_input_layernorm<half><<<grid, block, 0, stream>>>(
        out, input, bias, gamma, beta, m, n);
}

template <typename T>
__global__ void add_bias_input_layernorm_2(const T* __restrict input,
                                           const T* __restrict gamma,
                                           const T* __restrict beta,
                                           const T* __restrict bias,
                                           T* output,
                                           T* norm_output,
                                           int m,
                                           int n) {
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float local_out = (float)(__ldg(&input[blockIdx.x * n + i]));
    local_out += (float)(output[blockIdx.x * n + i]);
    local_out += (float)(__ldg(&bias[i]));
    output[blockIdx.x * n + i] = (T)local_out;
    local_sum += local_out;
  }

  mean = blockReduceSum<float>(local_sum);

  if (threadIdx.x == 0) s_mean = mean / n;
  __syncthreads();

  float local_var_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float diff = (float)(__ldg(&output[blockIdx.x * n + i])) - s_mean;
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum<float>(local_var_sum);

  if (threadIdx.x == 0) s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x) {
    norm_output[blockIdx.x * n + i] =
        (T)((((float)output[blockIdx.x * n + i] - s_mean) * s_variance) *
                (float)(__ldg(&gamma[i])) +
            (float)(__ldg(&beta[i])));
  }
}

template <typename T>
void add_bias_input_layernorm_2_kernelLauncher(const T* input,
                                               const T* gamma,
                                               const T* beta,
                                               const T* bias,
                                               T* output,
                                               T* norm_output,
                                               int m,
                                               int n,
                                               cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
  Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */

  if (n % 32 != 0) block.x = 1024;

  block.x =
      block.x / (4 / sizeof(T));  // if using half, only need half of block.x

  /* should pay attention to the rsqrt precision*/
  add_bias_input_layernorm_2<T><<<grid, block, 0, stream>>>(
      input, gamma, beta, bias, output, norm_output, m, n);  // For gpt-3
}

template <typename T>
__global__ void add_bias_input(
    T* output, const T* input, const T* bias, const int m, const int n) {
  // This kernel can run with any block size and grid size
  // Since the hidden dimension of GPT-3 would be larger than 1024
  const int bid = blockIdx.x;
  // const int blocks_per_row = n / blockDim.x;
  // const int col_index = (bid % blocks_per_row) * blockDim.x + threadIdx.x;
  // T bias_val = __ldg(&bias[col_index]);
  for (int index = bid * blockDim.x + threadIdx.x; index < m * n;
       index += blockDim.x * gridDim.x) {
    T bias_val = __ldg(&bias[index % n]);
    output[index] = output[index] + input[index] + bias_val;
  }
}

template <typename T>
void add_bias_input_kernelLauncher(T* output,
                                   const T* bias,
                                   const T* input,
                                   const int m,
                                   const int n,
                                   cudaStream_t stream) {
  int blocks_per_row = ceil(float(n) / 1024);
  dim3 grid(min(m * blocks_per_row, 65536));
  dim3 block(min(n, 1024));

  add_bias_input<<<grid, block, 0, stream>>>(output, input, bias, m, n);
}

template <typename T>
__global__ void layer_norm_kernel_generalize(const T* __restrict input,
                                             const T* __restrict gamma,
                                             const T* __restrict beta,
                                             T* output,
                                             int m,
                                             int n) {
  const int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    local_sum += (float)(__ldg(&input[blockIdx.x * n + i]));
  }

  mean = blockReduceSum<float>(local_sum);

  if (threadIdx.x == 0) s_mean = mean / n;
  __syncthreads();

  float local_var_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float diff = (float)(__ldg(&input[blockIdx.x * n + i])) - s_mean;
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum<float>(local_var_sum);

  if (threadIdx.x == 0) s_variance = rsqrtf(variance / n + 1e-6);

  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x) {
    output[blockIdx.x * n + i] =
        (T)((((float)input[blockIdx.x * n + i] - s_mean) * s_variance) *
                (float)(__ldg(&gamma[i])) +
            (float)(__ldg(&beta[i])));
  }
}

template <typename T>
void layer_norm(const T* input,
                const T* gamma,
                const T* beta,
                T* output,
                int m,
                int n,
                cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
     Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */
  if (n % 32 != 0) block.x = 1024;

  block.x =
      block.x / (4 / sizeof(T));  // if using half, only need half of block.x

  /* should pay attention to the rsqrt precision*/
  layer_norm_kernel_generalize<T><<<grid, block, 0, stream>>>(
      input, gamma, beta, output, m, n);  // For gpt-3
}

template void add_bias_act_kernelLauncher<float>(float* out,
                                                 const float* bias,
                                                 int m,
                                                 int n,
                                                 ActivationType activation_type,
                                                 cudaStream_t stream);

template void add_bias_input_layernorm_kernelLauncher<float>(
    float* out,
    const float* input,
    const float* bias,
    const float* gamma,
    const float* beta,
    int m,
    int n,
    cudaStream_t stream);

template void add_bias_act_kernelLauncher<half>(half* out,
                                                const half* bias,
                                                int m,
                                                int n,
                                                ActivationType activation_type,
                                                cudaStream_t stream);

template void add_bias_input_layernorm_kernelLauncher<half>(
    half* out,
    const half* input,
    const half* bias,
    const half* gamma,
    const half* beta,
    int m,
    int n,
    cudaStream_t stream);

template void add_bias_input_layernorm_2_kernelLauncher<float>(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* bias,
    float* output,
    float* norm_output,
    int m,
    int n,
    cudaStream_t stream);

template void add_bias_input_layernorm_2_kernelLauncher<half>(
    const half* input,
    const half* gamma,
    const half* beta,
    const half* bias,
    half* output,
    half* norm_output,
    int m,
    int n,
    cudaStream_t stream);

template void add_bias_input_kernelLauncher<float>(float* output,
                                                   const float* bias,
                                                   const float* input,
                                                   const int m,
                                                   const int n,
                                                   cudaStream_t stream);

template void add_bias_input_kernelLauncher<half>(half* output,
                                                  const half* bias,
                                                  const half* input,
                                                  const int m,
                                                  const int n,
                                                  cudaStream_t stream);

template void layer_norm<float>(const float* input,
                                const float* gamma,
                                const float* beta,
                                float* output,
                                int m,
                                int n,
                                cudaStream_t stream);

template void layer_norm<half>(const half* input,
                               const half* gamma,
                               const half* beta,
                               half* output,
                               int m,
                               int n,
                               cudaStream_t stream);

}  // namespace fastertransformer
