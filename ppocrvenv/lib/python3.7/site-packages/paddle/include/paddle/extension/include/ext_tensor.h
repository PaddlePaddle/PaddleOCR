/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <vector>
#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "ext_dll_decl.h"  // NOLINT
#include "ext_dtype.h"     // NOLINT
#include "ext_place.h"     // NOLINT

namespace paddle {
namespace framework {
class CustomTensorUtils;
}  // namespace framework

class StreamWrapper {
 public:
  StreamWrapper() : stream_(nullptr), is_stream_set_(false) {}
  void SetStream(void* stream) {
    stream_ = stream;
    is_stream_set_ = true;
  }

  void* GetStream() const { return stream_; }

  bool IsStreamSet() const { return is_stream_set_; }

 private:
  //  cudaStream_t stream_;
  void* stream_;
  bool is_stream_set_;
};

class PD_DLL_DECL Tensor {
 public:
  /// \brief Construct a Tensor on target Place for CustomOp.
  /// Generally it's only used for user to create Tensor.
  explicit Tensor(const PlaceType& place);
  /// \brief Construct a Tensor on target Place with shape for CustomOp.
  /// Generally it's only used for user to create Tensor.
  Tensor(const PlaceType& place, const std::vector<int64_t>& shape);
  /// \brief Reset the shape of the tensor.
  /// Generally it's only used for the input tensor.
  /// Reshape must be called before calling
  /// mutable_data() or copy_to(const PlaceType& place)
  /// \param shape The shape to set.
  void reshape(const std::vector<int64_t>& shape);

  /// \brief Get the memory pointer in CPU or GPU with
  /// specific data type.
  /// Please Reshape the tensor first before call this.
  /// It's usually used to get input data pointer.
  /// \param place The place of the tensor this will
  /// override the original place of current tensor.
  template <typename T>
  T* mutable_data(const PlaceType& place);

  /// \brief Get the memory pointer in CPU or GPU with
  /// specific data type. Please Reshape the tensor
  /// first before call this.It's usually used to get
  /// input data pointer.
  template <typename T>
  T* mutable_data();

  /// \brief Get the memory pointer directly.
  /// It's usually used to get the output data pointer.
  /// \return The tensor data buffer pointer.
  template <typename T>
  T* data() const;

  /// \brief Copy the host memory to tensor data.
  /// It's usually used to set the input tensor data.
  /// \param PlaceType of target place, of which
  /// the tensor will copy to.
  template <typename T>
  Tensor copy_to(const PlaceType& place) const;

  /// \brief Return a sub-tensor of the given tensor.
  /// It is usually used to extract a sub-tensor (which supports
  /// modifying the data of the original tensor) to perform further
  /// operations.
  /// \param begin_idx The index of the start row (inclusive) to slice.
  ///                  The index number begins from 0.
  /// \param end_idx  The index of the end row (exclusive) to slice.
  ///                 The index number begins from begin_idx + 1.
  /// \return The sliced tensor.
  Tensor slice(const int64_t begin_idx, const int64_t end_idx) const;

  /// \brief Return the shape of the Tensor.
  std::vector<int64_t> shape() const;

  /// \brief Return the data type of the tensor.
  /// It's usually used to get the output tensor data type.
  /// \return The data type of the tensor.
  DataType type() const;

  /// \brief Get the size of current tensor.
  /// Use this method to get the size of tensor
  /// \return int64_t.
  int64_t size() const;

  /// \brief Get the place of current tensor.
  /// Use this method to get the place of tensor
  /// \return Place.
  const PlaceType& place() const;

  /// \brief Cast datatype from one to another
  Tensor cast(const DataType& target_type) const;

  /// \brief Check Tensor is initialized
  bool is_initialized() const;

#if defined(PADDLE_WITH_CUDA)
  /// \bref Get current stream of Tensor
  cudaStream_t stream() const;
#elif defined(PADDLE_WITH_HIP)
  hipStream_t stream() const;
#endif

 private:
  friend class framework::CustomTensorUtils;
  mutable std::shared_ptr<void> tensor_;
  mutable PlaceType place_;
  StreamWrapper stream_;
};

}  // namespace paddle
