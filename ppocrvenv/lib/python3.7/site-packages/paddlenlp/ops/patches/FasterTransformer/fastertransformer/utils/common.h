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
#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include "stdio.h"

#define MAX_CONFIG_NUM 20
#define GEMM_NUM 6
#define COL32_ 32
#define ACTIVATION_AMAX_NUM 80
#define INT8O_GEMM_NUM 8
#define TRT_FUSED_MHA_AMAX_NUM 3
#define GEMM_CONFIG "gemm_config.in"
#define IGEMM_CONFIG "igemm_config.in"
// workspace for cublas gemm : 32MB
#define CUBLAS_WORKSPACE_SIZE 33554432


#include "fastertransformer/gemm_test/encoder_gemm_func.h"
#include "fastertransformer/gemm_test/encoder_igemm_func.h"

struct AbstractParam {
  virtual ~AbstractParam(){};
};

namespace fastertransformer {

enum { FLOAT_DATATYPE = 0, HALF_DATATYPE = 1, INT8_DATATYPE = 2 };

enum class OperationType { FP32, FP16 };
enum class AllocatorType { CUDA, PD };

#define PRINT_FUNC_NAME_()                                          \
  do {                                                              \
    std::cout << "[FT][CALL] " << __FUNCTION__ << " " << std::endl; \
  } while (0)

static double diffTime(timeval start, timeval end) {
  return (end.tv_sec - start.tv_sec) * 1000 +
         (end.tv_usec - start.tv_usec) * 0.001;
}

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorString(error);
}

static inline __device__ int8_t float_to_int8_rn(float x) {
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t &>(dst);
}

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

template <typename T>
void check(T result,
           char const *const func,
           const char *const file,
           int const line) {
  if (result) {
    throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") +
                             (_cudaGetErrorEnum(result)) + " " + file + ":" +
                             std::to_string(line) + " \n");
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void print_to_file(const T *result, const int size, const char *file) {
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
  printf("[INFO] file: %s \n", file);
  FILE *fd = fopen(file, "w");
  T *tmp = (T *)malloc(sizeof(T) * size);
  check_cuda_error(
      cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost));
  for (int i = 0; i < size; ++i) {
    float val;
    if (sizeof(T) == 2)
      val = (T)__half2float(tmp[i]);
    else
      val = (T)tmp[i];
    fprintf(fd, "%f\n", val);
  }
  free(tmp);
  fclose(fd);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
}

template <typename T>
void print_to_file(const T *result,
                   const int size,
                   const char *file,
                   cudaStream_t stream,
                   std::ios::openmode open_mode = std::ios::out) {
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
  printf("[INFO] file: %s with size %d.\n", file, size);
  std::ofstream outFile(file, open_mode);
  if (outFile) {
    T *tmp = new T[size];
    check_cuda_error(cudaMemcpyAsync(
        tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost, stream));
    for (int i = 0; i < size; ++i) {
      float val;
      if (sizeof(T) == 2)
        val = (T)__half2float(tmp[i]);
      else
        val = (T)tmp[i];
      outFile << val << std::endl;
    }
    delete[] tmp;
  } else {
    printf("[ERROR] cannot open file %s \n", file);
    exit(-1);
  }
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
}

template <typename T>
void print_to_screen(T *result, const int size) {
  float *tmp = (float *)malloc(sizeof(float) * size);
  check_cuda_error(
      cudaMemcpy(tmp, result, sizeof(float) * size, cudaMemcpyDeviceToHost));
  for (int i = 0; i < size; ++i) printf("%d, %f\n", i, (float)tmp[i]);
  free(tmp);
}

template <typename T>
void check_max_val(const T *result, const int size) {
  T *tmp = new T[size];
  cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost);
  float max_val = -100000;
  for (int i = 0; i < size; i++) {
    float val = (float)(tmp[i]);
    if (val > max_val) max_val = val;
  }
  delete tmp;
  printf("[INFO][CUDA] addr %p max val: %f \n", result, max_val);
}

inline int getSMVersion() {
  int device{-1};
  check_cuda_error(cudaGetDevice(&device));
  cudaDeviceProp props;
  check_cuda_error(cudaGetDeviceProperties(&props, device));
  return props.major * 10 + props.minor;
}

template <typename T>
void check_abs_mean_val(const T *result, const int size) {
  T *tmp = new T[size];
  cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost);
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    sum += abs((float)tmp[i]);
  }
  delete tmp;
  printf("[INFO][CUDA] addr %p abs mean val: %f \n", result, sum / size);
}

inline int div_up(int a, int n) { return (a + n - 1) / n; }

inline void print_mem_usage() {
  size_t free_bytes, total_bytes;
  check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
  float free = (float)(free_bytes) / 1024.0 / 1024.0 / 1024.0;
  float total = (float)(total_bytes) / 1024.0 / 1024.0 / 1024.0;
  printf("after allocation, free %.2f GB total %.2f GB\n", free, total);
}

}  // namespace fastertransformer
