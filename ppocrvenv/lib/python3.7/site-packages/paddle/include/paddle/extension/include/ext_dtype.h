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

#include <cstdint>
#include <string>

#include "complex.h"        // NOLINT
#include "ext_exception.h"  // NOLINT
#include "float16.h"        // NOLINT

namespace paddle {

using complex64 = paddle::platform::complex<float>;
using complex128 = paddle::platform::complex<double>;
using float16 = paddle::platform::float16;

enum class DataType {
  BOOL,
  INT8,
  UINT8,
  INT16,
  INT32,
  INT64,
  FLOAT16,
  FLOAT32,
  FLOAT64,
  COMPLEX64,
  COMPLEX128,
  // TODO(JiabinYang) support more data types if needed.
};

inline std::string ToString(DataType dtype) {
  switch (dtype) {
    case DataType::BOOL:
      return "bool";
    case DataType::INT8:
      return "int8_t";
    case DataType::UINT8:
      return "uint8_t";
    case DataType::INT16:
      return "int16_t";
    case DataType::INT32:
      return "int32_t";
    case DataType::INT64:
      return "int64_t";
    case DataType::FLOAT16:
      return "float16";
    case DataType::FLOAT32:
      return "float";
    case DataType::FLOAT64:
      return "double";
    case DataType::COMPLEX64:
      return "complex64";
    case DataType::COMPLEX128:
      return "complex128";
    default:
      PD_THROW("Unsupported paddle enum data type.");
  }
}

#define PD_FOR_EACH_DATA_TYPE(_)    \
  _(bool, DataType::BOOL)           \
  _(int8_t, DataType::INT8)         \
  _(uint8_t, DataType::UINT8)       \
  _(int16_t, DataType::INT16)       \
  _(int, DataType::INT32)           \
  _(int64_t, DataType::INT64)       \
  _(float16, DataType::FLOAT16)     \
  _(float, DataType::FLOAT32)       \
  _(double, DataType::FLOAT64)      \
  _(complex64, DataType::COMPLEX64) \
  _(complex128, DataType::COMPLEX128)

template <paddle::DataType T>
struct DataTypeToCPPType;

#define PD_SPECIALIZE_DataTypeToCPPType(cpp_type, data_type) \
  template <>                                                \
  struct DataTypeToCPPType<data_type> {                      \
    using type = cpp_type;                                   \
  };

PD_FOR_EACH_DATA_TYPE(PD_SPECIALIZE_DataTypeToCPPType)

#undef PD_SPECIALIZE_DataTypeToCPPType

}  // namespace paddle
