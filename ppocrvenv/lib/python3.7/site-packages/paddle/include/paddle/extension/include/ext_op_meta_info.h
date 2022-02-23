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

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "any.h"
#include "ext_dll_decl.h"   // NOLINT
#include "ext_exception.h"  // NOLINT
#include "ext_tensor.h"     // NOLINT

/**
 * Op Meta Info Related Define.
 *
 * Used to maintain operator core information.
 *
 */

namespace paddle {
namespace framework {
class PD_DLL_DECL OpMetaInfoHelper;
}  // namespace framework

using Tensor = paddle::Tensor;

///////////////// Util Marco Define ////////////////

#define PD_DISABLE_COPY_AND_ASSIGN(classname)      \
 private:                                          \
  classname(const classname&) = delete;            \
  classname(classname&&) = delete;                 \
  classname& operator=(const classname&) = delete; \
  classname& operator=(classname&&) = delete

#define STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)                        \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

///////////////// Util Define and Function ////////////////

constexpr char kGradTensorSuffix[] = "@GRAD";
constexpr char kTensorVectorSuffix[] = "@VECTOR";

// Used for Construct Grad Tensor name
inline std::string Grad(const std::string& t_name) {
  std::string result;
  result.reserve(t_name.size() + 5U);
  result += t_name;
  result += kGradTensorSuffix;
  return result;
}

// Used for Construct std::vector<Tensor> name
inline std::string Vec(const std::string& t_name) {
  std::string result;
  result.reserve(t_name.size() + 7U);
  result += t_name;
  result += kTensorVectorSuffix;
  return result;
}

////////////////////// Kernel Function (PD_KERNEL) ////////////////////////

// Record Op kernel core function
using KernelFunc =
    std::vector<Tensor> (*)(const std::vector<Tensor>& inputs,
                            const std::vector<std::vector<Tensor>>& vec_inputs,
                            const std::vector<paddle::any>& attrs);

#define PD_SPECIALIZE_ComputeCallHelper(attr_type)                            \
  template <typename... Tail>                                                 \
  struct ComputeCallHelper<attr_type, Tail...> {                              \
    template <int in_idx, int vec_in_idx, int attr_idx,                       \
              typename... PreviousArgs>                                       \
    static Return Compute(const std::vector<Tensor>& inputs,                  \
                          const std::vector<std::vector<Tensor>>& vec_inputs, \
                          const std::vector<paddle::any>& attrs,              \
                          const PreviousArgs&... pargs) {                     \
      try {                                                                   \
        attr_type arg = paddle::any_cast<attr_type>(attrs[attr_idx]);         \
        return ComputeCallHelper<Tail...>::template Compute<                  \
            in_idx, vec_in_idx, attr_idx + 1>(inputs, vec_inputs, attrs,      \
                                              pargs..., arg);                 \
      } catch (paddle::bad_any_cast&) {                                       \
        PD_THROW(                                                             \
            "Attribute cast error in custom operator. Expected " #attr_type   \
            " value.");                                                       \
      }                                                                       \
    }                                                                         \
  }

template <typename T>
struct TypeTag {};

template <typename F, F f>
struct KernelFuncImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct KernelFuncImpl<Return (*)(Args...), impl_fn> {
  static Return Compute(const std::vector<Tensor>& inputs,
                        const std::vector<std::vector<Tensor>>& vec_inputs,
                        const std::vector<paddle::any>& attrs) {
    return ComputeCallHelper<Args..., TypeTag<int>>::template Compute<0, 0, 0>(
        inputs, vec_inputs, attrs);
  }

 private:
  template <typename... RemainingArgs>
  struct ComputeCallHelper;

  template <typename... Tail>
  struct ComputeCallHelper<const Tensor&, Tail...> {
    template <int in_idx, int vec_in_idx, int attr_idx,
              typename... PreviousArgs>
    static Return Compute(const std::vector<Tensor>& inputs,
                          const std::vector<std::vector<Tensor>>& vec_inputs,
                          const std::vector<paddle::any>& attrs,
                          const PreviousArgs&... pargs) {
      const Tensor& arg = inputs[in_idx];
      return ComputeCallHelper<Tail...>::template Compute<in_idx + 1,
                                                          vec_in_idx, attr_idx>(
          inputs, vec_inputs, attrs, pargs..., arg);
    }
  };

  template <typename... Tail>
  struct ComputeCallHelper<const std::vector<Tensor>&, Tail...> {
    template <int in_idx, int vec_in_idx, int attr_idx,
              typename... PreviousArgs>
    static Return Compute(const std::vector<Tensor>& inputs,
                          const std::vector<std::vector<Tensor>>& vec_inputs,
                          const std::vector<paddle::any>& attrs,
                          const PreviousArgs&... pargs) {
      const std::vector<Tensor>& arg = vec_inputs[vec_in_idx];
      return ComputeCallHelper<Tail...>::template Compute<
          in_idx, vec_in_idx + 1, attr_idx>(inputs, vec_inputs, attrs, pargs...,
                                            arg);
    }
  };

  PD_SPECIALIZE_ComputeCallHelper(const bool&);
  PD_SPECIALIZE_ComputeCallHelper(const int&);
  PD_SPECIALIZE_ComputeCallHelper(const float&);
  PD_SPECIALIZE_ComputeCallHelper(const int64_t&);
  PD_SPECIALIZE_ComputeCallHelper(const std::string&);
  PD_SPECIALIZE_ComputeCallHelper(const std::vector<int>&);
  PD_SPECIALIZE_ComputeCallHelper(const std::vector<float>&);
  PD_SPECIALIZE_ComputeCallHelper(const std::vector<int64_t>&);
  PD_SPECIALIZE_ComputeCallHelper(const std::vector<std::string>&);
  // TODO(chenweihang): support other attribute type if needed.
  // Why not support other attribute type here?
  // - boost::blank, std::vector<bool> and std::vector<double>
  //   are not used in op
  // - BlockDesc* and std::vector<BlockDesc*> are used in framework

  // NOTE(chenweihang): Used to be compatible with the 2.0.1 released
  // interface, and will be deprecated in the future
  PD_SPECIALIZE_ComputeCallHelper(bool);
  PD_SPECIALIZE_ComputeCallHelper(int);
  PD_SPECIALIZE_ComputeCallHelper(float);
  PD_SPECIALIZE_ComputeCallHelper(int64_t);
  PD_SPECIALIZE_ComputeCallHelper(std::string);
  PD_SPECIALIZE_ComputeCallHelper(std::vector<int>);
  PD_SPECIALIZE_ComputeCallHelper(std::vector<float>);
  PD_SPECIALIZE_ComputeCallHelper(std::vector<int64_t>);
  PD_SPECIALIZE_ComputeCallHelper(std::vector<std::string>);

  // end: base template
  template <typename T>
  struct ComputeCallHelper<TypeTag<T>> {
    template <int in_idx, int vec_in_idx, int attr_idx>
    static Return Compute(const std::vector<Tensor>& inputs,
                          const std::vector<std::vector<Tensor>>& vec_inputs,
                          const std::vector<paddle::any>& attrs,
                          const Args&... args) {
      return impl_fn(args...);
    }
  };
};

#define PD_KERNEL(...) \
  ::paddle::KernelFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute

/////////////// InferShape Function (PD_INFER_SHAPE) ///////////////

// Record Op infershape core function
using InferShapeFunc = std::vector<std::vector<int64_t>> (*)(
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::vector<std::vector<int64_t>>>& vec_input_shapes,
    const std::vector<paddle::any>& attrs);

#define PD_SPECIALIZE_InferShapeCallHelper_FOR_SHAPE(input_type)               \
  template <typename... Tail>                                                  \
  struct InferShapeCallHelper<input_type, Tail...> {                           \
    template <int in_idx, int vec_in_idx, int attr_idx,                        \
              typename... PreviousArgs>                                        \
    static Return InferShape(                                                  \
        const std::vector<std::vector<int64_t>>& input_shapes,                 \
        const std::vector<std::vector<std::vector<int64_t>>>&                  \
            vec_input_shapes,                                                  \
        const std::vector<paddle::any>& attrs, const PreviousArgs&... pargs) { \
      input_type arg = input_shapes[in_idx];                                   \
      return InferShapeCallHelper<Tail...>::template InferShape<               \
          in_idx + 1, vec_in_idx, attr_idx>(input_shapes, vec_input_shapes,    \
                                            attrs, pargs..., arg);             \
    }                                                                          \
  }

#define PD_SPECIALIZE_InferShapeCallHelper_FOR_SHAPES(input_type)              \
  template <typename... Tail>                                                  \
  struct InferShapeCallHelper<input_type, Tail...> {                           \
    template <int in_idx, int vec_in_idx, int attr_idx,                        \
              typename... PreviousArgs>                                        \
    static Return InferShape(                                                  \
        const std::vector<std::vector<int64_t>>& input_shapes,                 \
        const std::vector<std::vector<std::vector<int64_t>>>&                  \
            vec_input_shapes,                                                  \
        const std::vector<paddle::any>& attrs, const PreviousArgs&... pargs) { \
      input_type arg = vec_input_shapes[vec_in_idx];                           \
      return InferShapeCallHelper<Tail...>::template InferShape<               \
          in_idx, vec_in_idx + 1, attr_idx>(input_shapes, vec_input_shapes,    \
                                            attrs, pargs..., arg);             \
    }                                                                          \
  }

#define PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(attr_type)                 \
  template <typename... Tail>                                                  \
  struct InferShapeCallHelper<attr_type, Tail...> {                            \
    template <int in_idx, int vec_in_idx, int attr_idx,                        \
              typename... PreviousArgs>                                        \
    static Return InferShape(                                                  \
        const std::vector<std::vector<int64_t>>& input_shapes,                 \
        const std::vector<std::vector<std::vector<int64_t>>>&                  \
            vec_input_shapes,                                                  \
        const std::vector<paddle::any>& attrs, const PreviousArgs&... pargs) { \
      try {                                                                    \
        attr_type arg = paddle::any_cast<attr_type>(attrs[attr_idx]);          \
        return InferShapeCallHelper<Tail...>::template InferShape<             \
            in_idx, vec_in_idx, attr_idx + 1>(input_shapes, vec_input_shapes,  \
                                              attrs, pargs..., arg);           \
      } catch (paddle::bad_any_cast&) {                                        \
        PD_THROW(                                                              \
            "Attribute cast error in custom operator InferShapeFn. "           \
            "Expected " #attr_type                                             \
            " value. InferShapeFn's attribute list must be exactly same as "   \
            "Forward "                                                         \
            "KernelFn's attribute list except std::vector<int64_t> "           \
            "attribute.");                                                     \
      }                                                                        \
    }                                                                          \
  }

template <typename F, F f>
struct InferShapeFuncImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct InferShapeFuncImpl<Return (*)(Args...), impl_fn> {
  static Return InferShape(
      const std::vector<std::vector<int64_t>>& input_shapes,
      const std::vector<std::vector<std::vector<int64_t>>>& vec_input_shapes,
      const std::vector<paddle::any>& attrs) {
    return InferShapeCallHelper<Args..., TypeTag<int>>::template InferShape<
        0, 0, 0>(input_shapes, vec_input_shapes, attrs);
  }

 private:
  template <typename... RemainingArgs>
  struct InferShapeCallHelper;

  PD_SPECIALIZE_InferShapeCallHelper_FOR_SHAPE(const std::vector<int64_t>&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_SHAPES(
      const std::vector<std::vector<int64_t>>&);

  // NOTE(chenweihang): Used to be compatible with the 2.0.1 released
  // interface, and will be deprecated in the future
  PD_SPECIALIZE_InferShapeCallHelper_FOR_SHAPE(std::vector<int64_t>);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_SHAPES(
      std::vector<std::vector<int64_t>>);

  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const bool&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const int&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const float&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const int64_t&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const std::string&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const std::vector<int>&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const std::vector<float>&);
  PD_SPECIALIZE_InferShapeCallHelper_FOR_ATTR(const std::vector<std::string>&);
  // NOTE(chenweihang): InferShape can't support std::vector<int64_t> attr type,
  // because the input type is std::vector<int64_t>, only can use one rule to
  // parse std::vector<int64_t> parameter

  // end: base template
  template <typename T>
  struct InferShapeCallHelper<TypeTag<T>> {
    template <int in_idx, int vec_in_idx, int attr_idx>
    static Return InferShape(
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<std::vector<std::vector<int64_t>>>& vec_input_shapes,
        const std::vector<paddle::any>& attrs, const Args&... args) {
      return impl_fn(args...);
    }
  };
};

#define PD_INFER_SHAPE(...) \
  ::paddle::InferShapeFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::InferShape

/////////////// InferDataType Function (PD_INFER_DTYPE) ///////////////

// Record Op Infer dtype core function
using InferDtypeFunc = std::vector<DataType> (*)(
    const std::vector<DataType>& input_dtypes,
    const std::vector<std::vector<DataType>>& vec_input_dtypes);

#define PD_SPECIALIZE_InferDtypeCallHelper_TO_DTYPE(input_type)              \
  template <typename... Tail>                                                \
  struct InferDtypeCallHelper<input_type, Tail...> {                         \
    template <int in_idx, int vec_in_idx, typename... PreviousArgs>          \
    static Return InferDtype(                                                \
        const std::vector<DataType>& input_dtypes,                           \
        const std::vector<std::vector<DataType>>& vec_input_dtypes,          \
        const PreviousArgs&... pargs) {                                      \
      input_type arg = input_dtypes[in_idx];                                 \
      return InferDtypeCallHelper<Tail...>::template InferDtype<in_idx + 1,  \
                                                                vec_in_idx>( \
          input_dtypes, vec_input_dtypes, pargs..., arg);                    \
    }                                                                        \
  }

#define PD_SPECIALIZE_InferDtypeCallHelper_FOR_DTYPES(input_type)           \
  template <typename... Tail>                                               \
  struct InferDtypeCallHelper<input_type, Tail...> {                        \
    template <int in_idx, int vec_in_idx, typename... PreviousArgs>         \
    static Return InferDtype(                                               \
        const std::vector<DataType>& input_dtypes,                          \
        const std::vector<std::vector<DataType>>& vec_input_dtypes,         \
        const PreviousArgs&... pargs) {                                     \
      input_type arg = vec_input_dtypes[vec_in_idx];                        \
      return InferDtypeCallHelper<Tail...>::template InferDtype<            \
          in_idx, vec_in_idx + 1>(input_dtypes, vec_input_dtypes, pargs..., \
                                  arg);                                     \
    }                                                                       \
  }

template <typename F, F f>
struct InferDtypeFuncImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct InferDtypeFuncImpl<Return (*)(Args...), impl_fn> {
  static Return InferDtype(
      const std::vector<DataType>& input_dtypes,
      const std::vector<std::vector<DataType>>& vec_input_dtypes) {
    return InferDtypeCallHelper<Args..., TypeTag<int>>::template InferDtype<0,
                                                                            0>(
        input_dtypes, vec_input_dtypes);
  }

 private:
  template <typename... RemainingArgs>
  struct InferDtypeCallHelper;

  PD_SPECIALIZE_InferDtypeCallHelper_TO_DTYPE(const DataType&);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_DTYPES(const std::vector<DataType>&);

  // NOTE(chenweihang): Used to be compatible with the 2.0.1 released
  // interface, and will be deprecated in the future
  PD_SPECIALIZE_InferDtypeCallHelper_TO_DTYPE(DataType);
  PD_SPECIALIZE_InferDtypeCallHelper_FOR_DTYPES(std::vector<DataType>);

  // end: base template
  template <typename T>
  struct InferDtypeCallHelper<TypeTag<T>> {
    template <int in_idx, int vec_in_idx>
    static Return InferDtype(
        const std::vector<DataType>& input_dtypes,
        const std::vector<std::vector<DataType>>& vec_input_dtypes,
        const Args&... args) {
      return impl_fn(args...);
    }
  };
};

#define PD_INFER_DTYPE(...) \
  ::paddle::InferDtypeFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::InferDtype

////////////////////// Op Meta Info //////////////////////

class PD_DLL_DECL OpMetaInfo {
 public:
  explicit OpMetaInfo(const std::string& op_name) : name_(op_name) {}

  // format: {"<name1>", "<name2>", ...}
  OpMetaInfo& Inputs(std::vector<std::string>&& inputs);

  // format: {"<name1>", "<name2>", ...}
  OpMetaInfo& Outputs(std::vector<std::string>&& outputs);

  // format: {"<name1>:<type1>", "<name1>:<type1>", ...}
  OpMetaInfo& Attrs(std::vector<std::string>&& attrs);

  // format: PD_KERNEL(...)
  OpMetaInfo& SetKernelFn(KernelFunc&& func);

  // format: PD_INFER_SHAPE(...)
  OpMetaInfo& SetInferShapeFn(InferShapeFunc&& func);

  // format: PD_INFER_DTYPE(...)
  OpMetaInfo& SetInferDtypeFn(InferDtypeFunc&& func);

 private:
  friend class framework::OpMetaInfoHelper;

  // 1. desc info
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  std::vector<std::string> attrs_;

  // 2. func info
  KernelFunc kernel_fn_{nullptr};
  InferShapeFunc infer_shape_fn_{nullptr};
  InferDtypeFunc infer_dtype_fn_{nullptr};
};

//////////////// Op Meta Info Map /////////////////

class PD_DLL_DECL OpMetaInfoMap {
 public:
  // this function's impl should keep in header file.
  // if move to cc file, meta info can not be added
  // into map
  static OpMetaInfoMap& Instance() {
    static OpMetaInfoMap g_custom_op_meta_info_map;
    return g_custom_op_meta_info_map;
  }

  std::vector<OpMetaInfo>& operator[](const std::string& name);

  const std::unordered_map<std::string, std::vector<OpMetaInfo>>& GetMap()
      const;

 private:
  OpMetaInfoMap() = default;
  std::unordered_map<std::string, std::vector<OpMetaInfo>> map_;

  PD_DISABLE_COPY_AND_ASSIGN(OpMetaInfoMap);
};

//////////////// Op Meta Info Builder /////////////////

class PD_DLL_DECL OpMetaInfoBuilder {
 public:
  explicit OpMetaInfoBuilder(std::string&& name, size_t index);
  OpMetaInfoBuilder& Inputs(std::vector<std::string>&& inputs);
  OpMetaInfoBuilder& Outputs(std::vector<std::string>&& outputs);
  OpMetaInfoBuilder& Attrs(std::vector<std::string>&& attrs);
  OpMetaInfoBuilder& SetKernelFn(KernelFunc func);
  OpMetaInfoBuilder& SetInferShapeFn(InferShapeFunc func);
  OpMetaInfoBuilder& SetInferDtypeFn(InferDtypeFunc func);

 private:
  // Forward Op name
  std::string name_;
  // ref current info ptr
  OpMetaInfo* info_ptr_;
  // The current op meta info index in vector
  // - 0: op, 1: grad_op, 2: grad_grad_op
  size_t index_;
};

/////////////////////// Op register API /////////////////////////

// For inference: compile directly with framework
// Call after PD_BUILD_OP(...)
void RegisterAllCustomOperator();

// Using this api to load compiled custom operator's dynamic library and
// register Custom
// Operator into it
void LoadCustomOperatorLib(const std::string& dso_name);

/////////////////////// Op register Macro /////////////////////////

#define PD_BUILD_OP(op_name)                                                   \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                              \
      __reg_op__##op_name, "PD_BUILD_OP must be called in global namespace."); \
  static ::paddle::OpMetaInfoBuilder __op_meta_info_##op_name##__ =            \
      ::paddle::OpMetaInfoBuilder(#op_name, 0)

#define PD_BUILD_GRAD_OP(op_name)                                        \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                        \
      __reg_grad_op__##op_name,                                          \
      "PD_BUILD_GRAD_OP must be called in global namespace.");           \
  static ::paddle::OpMetaInfoBuilder __grad_op_meta_info_##op_name##__ = \
      ::paddle::OpMetaInfoBuilder(#op_name, 1)

#define PD_BUILD_DOUBLE_GRAD_OP(op_name)                                      \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                             \
      __reg_grad_grad_op__##op_name,                                          \
      "PD_BUILD_DOUBLE_GRAD_OP must be called in global namespace.");         \
  static ::paddle::OpMetaInfoBuilder __grad_grad_op_meta_info_##op_name##__ = \
      ::paddle::OpMetaInfoBuilder(#op_name, 2)

}  // namespace paddle

///////////////////// C API ///////////////////

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
// C-API to get global OpMetaInfoMap.
__declspec(dllexport) inline paddle::OpMetaInfoMap& PD_GetOpMetaInfoMap() {
  return paddle::OpMetaInfoMap::Instance();
}
#endif  // _WIN32

#ifdef __cplusplus
}
#endif
