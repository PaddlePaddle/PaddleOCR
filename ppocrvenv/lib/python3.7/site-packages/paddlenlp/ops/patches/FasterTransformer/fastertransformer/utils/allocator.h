/*
 * Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
 * Memory Allocator
 **/

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "fastertransformer/utils/common.h"
#include "fastertransformer/utils/utils.h"

#ifdef PADDLE_CUDA
#ifdef PADDLE_ON_INFERENCE
#include "paddle/include/experimental/ext_all.h"
#else
#include "paddle/extension.h"
#endif
#endif

namespace fastertransformer {

class IAllocator {
public:
  virtual void *malloc(size_t size, const bool is_set_zero = true) const = 0;
  virtual void free(void *ptr) const = 0;
};

template <AllocatorType AllocType_>
class Allocator;

template <>
class Allocator<AllocatorType::CUDA> : public IAllocator {
  const int device_id_;

public:
  Allocator(int device_id) : device_id_(device_id) {}
  ~Allocator() {}

  void *malloc(size_t size, const bool is_set_zero = true) const {
    void *ptr = nullptr;
    int o_device = 0;
    check_cuda_error(get_set_device(device_id_, &o_device));
    check_cuda_error(cudaMalloc(&ptr, size));
    check_cuda_error(get_set_device(o_device));
    return ptr;
  }

  void free(void *ptr) const {
    int o_device = 0;
    check_cuda_error(get_set_device(device_id_, &o_device));
    check_cuda_error(cudaFree(ptr));
    check_cuda_error(get_set_device(o_device));
    return;
  }
};

#ifdef PADDLE_CUDA
template <>
class Allocator<AllocatorType::PD> : public IAllocator {
  std::shared_ptr<std::vector<paddle::Tensor>> allocated_tensor_vector;
  cudaStream_t stream_;

public:
  Allocator(cudaStream_t stream)
      : allocated_tensor_vector(
            std::make_shared<std::vector<paddle::Tensor>>()),
        stream_(stream) {}

  void *malloc(size_t size, const bool is_set_zero = true) const {
    int64_t buf_size = static_cast<int64_t>(size);
    std::vector<int64_t> buf_dims({buf_size});
    auto buf = paddle::Tensor(paddle::PlaceType::kGPU, buf_dims);
    allocated_tensor_vector->push_back(buf);

    auto *flat = buf.mutable_data<uint8_t>(paddle::PlaceType::kGPU);
    void *ptr = reinterpret_cast<void *>(flat);
    return ptr;
  }

  void free(void *ptr) const {
#ifndef NDEBUG
    printf("call from allocator free\n");
#endif
    return;
  }

  //   ~Allocator() {
  //     allocated_tensor_vector->clear();
  //     delete allocated_tensor_vector;
  //   }
};
#endif

}  // namespace fastertransformer
