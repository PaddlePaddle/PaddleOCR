# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(FasterTransformer LANGUAGES C CXX CUDA)

find_package(CUDA 10.1 REQUIRED)

INCLUDE(ExternalProject)

set(CXX_STD "14" CACHE STRING "C++ standard")

option(ON_INFER         "Compiled with inference. "                                 OFF)
option(WITH_GPU         "Compiled with GPU/CPU, default use CPU."                   ON)
option(WITH_MKL         "Compile with MKL. Only works when ON_INFER is ON."         ON)
option(USE_TENSORRT     "Compiled with TensorRT."                                   OFF)
option(WITH_TRANSFORMER "Compiled with Transformer."                                ON)
option(WITH_GPT         "Compiled with GPT."                                        ON)
option(WITH_UNIFIED     "Compiled with Unified Transformer."                        ON)
option(WITH_SP          "Compiled with sentencepiece. Only works when WITH_GPT and ON_INFER is ON." OFF)
option(WITH_DECODER     "Compile with Transformer Decoder"                          ON)
option(WITH_ENCODER     "Compile with Transformer Encoder"                          ON)
option(WITH_STATIC_LIB  "Compile static lib"                                        OFF)
option(WITH_BART        "Compile with BART"                                         ON)
option(WITH_MBART       "Compile with MBART"                                        ON)

if(NOT WITH_GPU)
  message(FATAL_ERROR "Faster transformer custom op doesn't support CPU. Please add the flag -DWITH_GPU=ON to use GPU. ")
endif()

list(APPEND decoding_op_files cublas_handle.cc)

if(WITH_TRANSFORMER)
  list(APPEND decoding_op_files fusion_decoding_op.cc fusion_decoding_op.cu fusion_force_decoding_op.cc fusion_force_decoding_op.cu)
endif()

if(WITH_GPT)
  list(APPEND decoding_op_files fusion_gpt_op.cc fusion_gpt_op.cu)
endif()

if(WITH_UNIFIED)
  list(APPEND decoding_op_files fusion_unified_decoding_op.cc fusion_unified_decoding_op.cu)
endif()

# TODO: Support encoder when ON_INFER
if(WITH_ENCODER AND NOT ON_INFER)
  list(APPEND decoding_op_files fusion_encoder_op.cc fusion_encoder_op.cu)
endif()

if(WITH_DECODER)
  list(APPEND decoding_op_files fusion_decoder_op.cc fusion_decoder_op.cu)
endif()

if(WITH_BART)
  list(APPEND decoding_op_files fusion_bart_decoding_op.cc fusion_bart_decoding_op.cu)
endif()

if(WITH_MBART)
  list(APPEND decoding_op_files fusion_mbart_decoding_op.cc fusion_mbart_decoding_op.cu)
endif()

if(NOT WITH_TRANSFORMER AND NOT WITH_GPT AND NOT WITH_DECODER AND NOT WITH_ENCODER AND NOT WITH_BART AND NOT WITH_MBART)
  message(FATAL_ERROR "-DWITH_TRANSFORMER=ON or/and -DWITH_GPT=ON or/and -DWITH_DECODER=ON or/and -DWITH_ENCODER=ON or/and -DWITH_BART=ON or/and -DWITH_MBART=ON must be set to use FasterTransformer. ")
endif()

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

# Setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")    
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")

######################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   detect_installed_gpus(out_variable)
function(detect_installed_gpus out_variable)
  if(NOT CUDA_gpu_detect_output)
    set(cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

    file(WRITE ${cufile} ""
      "#include \"stdio.h\"\n"
      "#include \"cuda.h\"\n"
      "#include \"cuda_runtime.h\"\n"
      "int main() {\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device) {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")

    execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}"
                    "--run" "${cufile}"
                    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                    RESULT_VARIABLE nvcc_res OUTPUT_VARIABLE nvcc_out
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(nvcc_res EQUAL 0)
      # Only use last item of nvcc_out (the last device's compute capability).
      string(REGEX REPLACE "\\." "" nvcc_out "${nvcc_out}")
      string(REGEX MATCHALL "[0-9()]+" nvcc_out "${nvcc_out}")
      list(GET nvcc_out -1 nvcc_out)
      set(CUDA_gpu_detect_output ${nvcc_out} CACHE INTERNAL "Returned GPU architetures from detect_installed_gpus tool" FORCE)
    endif()
  endif()

  if(NOT CUDA_gpu_detect_output)
    message(STATUS "Automatic GPU detection failed. Building for all known architectures.")
    set(${out_variable} ${paddle_known_gpu_archs} PARENT_SCOPE)
  else()
    set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
  endif()
endfunction()

if (NOT SM)
  # TODO(guosheng): Remove it if `GetCUDAComputeCapability` is exposed by paddle.
  # Currently, if `CUDA_gpu_detect_output` is not defined, use the detected arch.
  detect_installed_gpus(SM)
endif()

#[[
if (SM STREQUAL 80 OR
    SM STREQUAL 86 OR
    SM STREQUAL 70 OR
    SM STREQUAL 75 OR
    SM STREQUAL 61 OR
    SM STREQUAL 60)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM},code=\\\"sm_${SM},compute_${SM}\\\"")
  if (SM STREQUAL 70 OR SM STREQUAL 75 OR SM STREQUAL 80 OR SM STREQUAL 86)
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  endif()
message("-- Assign GPU architecture (sm=${SM})")

else()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                      -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                      -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                      ")

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")

message("-- Assign GPU architecture (sm=70,75)")
endif()
]]

set(SM_SETS 52 60 61 70 75 80)
set(USING_WMMA False)
set(FIND_SM False)

foreach(SM_NUM IN LISTS SM_SETS)
  string(FIND "${SM}" "${SM_NUM}" SM_POS)
  if(SM_POS GREATER -1)
    set(FIND_SM True)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM_NUM},code=\\\"sm_${SM_NUM},compute_${SM_NUM}\\\"")

    if (SM_NUM STREQUAL 70 OR SM_NUM STREQUAL 75 OR SM_NUM STREQUAL 80 OR SM_NUM STREQUAL 86)
      set(USING_WMMA True)
    endif()

    set(CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
    message("-- Assign GPU architecture (sm=${SM_NUM})")
  endif()
endforeach()

if(USING_WMMA STREQUAL True)
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  message("-- Use WMMA")
endif()

if(NOT (FIND_SM STREQUAL True))
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        ")

  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  if(BUILD_PYT)
    set(ENV{TORCH_CUDA_ARCH_LIST} "7.0;7.5;8.0")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
  message("-- Assign GPU architecture (sm=70,75,80)")
endif()

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD}")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include)

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

set(THIRD_PATH "third-party")
set(THIRD_PARTY_NAME "fastertransformer")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(external/boost)

set(OPS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/utils/allocator.h allocator_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/utils/allocator.h allocator_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/utils/common.h common_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/utils/common.h common_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/CMakeLists.txt cmakelists_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/CMakeLists.txt cmakelists_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/cuda/topk_kernels.cu topk_kernels_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/cuda/topk_kernels.cu topk_kernels_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/cuda/lightseq_kernels.cu lightseq_kernels_cu_src)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/cuda/open_decoder.cu open_decoder_cu_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/cuda/open_decoder.cu open_decoder_cu_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/open_decoder.h open_decoder_h_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/open_decoder.h open_decoder_h_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/cuda/cuda_kernels.h cuda_kernels_h_src)
#file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/cuda/transformer_cuda_kernels.h trans_cuda_kernels_h_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/cuda/cuda_kernels.h cuda_kernels_h_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/cuda/cuda_kernels.cu cuda_kernels_cu_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/cuda/cuda_kernels.cu cuda_kernels_cu_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/cuda/decoding_kernels.cu decoding_kernels_cu_src)
file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/cuda/transformer_decoding_kernels.cu trans_decoding_kernels_cu_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/cuda/decoding_kernels.cu decoding_kernels_cu_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/cuda/open_decoder.cuh open_decoder_cuh_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/cuda/open_decoder.cuh open_decoder_cuh_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/utils/arguments.h arguments_h_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/utils/arguments.h arguments_h_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/decoding_beamsearch.h decoding_beamsearch_h_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/decoding_beamsearch.h decoding_beamsearch_h_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/decoding_sampling.h decoding_sampling_h_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/decoding_sampling.h decoding_sampling_h_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/cuda/online_softmax_beamsearch_kernels.cu online_softmax_beamsearch_kernels_cu_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/cuda/online_softmax_beamsearch_kernels.cu online_softmax_beamsearch_kernels_cu_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/cuda/topk_kernels.cuh topk_kernels_cuh_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/cuda/topk_kernels.cuh topk_kernels_cuh_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/cuda/transformer_kernels.cu trans_kernels_cu_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/cuda/transformer_kernels.cu trans_kernels_cu_dst)

# Encoder patches.
file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/bert_encoder_transformer.h bert_encoder_transformer_h_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/bert_encoder_transformer.h bert_encoder_transformer_h_dst)
file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/fastertransformer/standard_encoder.h standard_encoder_h_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/fastertransformer/standard_encoder.h standard_encoder_h_dst)
# Encoder patches end.

# TODO(guosheng): `find` seems meeting errors missing argument to `-exec', fix it
set(MUTE_COMMAND grep -rl "printf(\"\\[WARNING\\]" ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/ | xargs -i{} sed -i "s/printf(\"\\WWARNING\\W decoding[^)]\\{1,\\})/ /" {})
set(OPEN_ATTENTION_MUTE_COMMAND grep -rl "printf(\"\\[WARNING\\]" ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/${THIRD_PARTY_NAME}/ | xargs -i{} sed -i "s/printf(\"\\WWARNING\\W\\WOpenMultiHeadAttention\\W[^)]\\{1,\\})/ /" {})

set(FT_PATCH_COMMAND
  printf \\n\\n > blank_lines
  && cp ${allocator_src} ${allocator_dst}
  && cp ${common_src} ${common_dst}
  && cp ${cmakelists_src} ${cmakelists_dst}
  && cp ${topk_kernels_src} ${topk_kernels_dst}
  && cp ${decoding_beamsearch_h_src} ${decoding_beamsearch_h_dst}
  && cp ${decoding_sampling_h_src} ${decoding_sampling_h_dst}
  && cp ${online_softmax_beamsearch_kernels_cu_src} ${online_softmax_beamsearch_kernels_cu_dst}
  && cp ${arguments_h_src} ${arguments_h_dst}
  && cp ${open_decoder_h_src} ${open_decoder_h_dst}
  && cp ${standard_encoder_h_src} ${standard_encoder_h_dst}
  && cp ${bert_encoder_transformer_h_src} ${bert_encoder_transformer_h_dst}
  && cp ${trans_kernels_cu_src} ${trans_kernels_cu_dst}
  && cat blank_lines ${cuda_kernels_h_src} >> ${cuda_kernels_h_dst}
  && cat blank_lines ${lightseq_kernels_cu_src} >> ${topk_kernels_dst}
  && cat blank_lines ${cuda_kernels_cu_src} >> ${cuda_kernels_cu_dst}
  && cat blank_lines ${decoding_kernels_cu_src} >> ${decoding_kernels_cu_dst}
  && cat blank_lines ${topk_kernels_cuh_src} >> ${topk_kernels_cuh_dst}
  && cat blank_lines ${trans_decoding_kernels_cu_src} >> ${decoding_kernels_cu_dst}
  && cat blank_lines ${open_decoder_cu_src} >> ${open_decoder_cu_dst}
  && cat blank_lines ${open_decoder_cuh_src} >> ${open_decoder_cuh_dst}
  && sed -i "s/^#define NEW_TRANSPOSE_BATCH_MAJOR 1/#define NEW_TRANSPOSE_BATCH_MAJOR 0/g" ${open_decoder_cu_dst}
  && ${MUTE_COMMAND}
  && ${OPEN_ATTENTION_MUTE_COMMAND}
)

# TODO(guosheng): Use UPDATE_COMMAND instead of PATCH_COMMAND to make cmake
# re-run always use the latest patches when the developer changes FT patch codes,
# all patches rather than the changes would re-build, any better way to do this.
# Or maybe hidden this function for simplicity.
set(FT_UPDATE_COMMAND git checkout nccl_dependent_refine && git checkout . && ${FT_PATCH_COMMAND})

ExternalProject_Add(
  extern_${THIRD_PARTY_NAME}
  GIT_REPOSITORY    https://gitee.com/paddlepaddle/FasterTransformer.git
  GIT_TAG           nccl_dependent_refine
  PREFIX            ${THIRD_PATH}
  SOURCE_DIR        ${THIRD_PATH}/source/${THIRD_PARTY_NAME}
  UPDATE_COMMAND    ${FT_UPDATE_COMMAND}  # PATCH_COMMAND     ${FT_PATCH_COMMAND}
  BINARY_DIR        ${THIRD_PATH}/build/${THIRD_PARTY_NAME}
  INSTALL_COMMAND   ""
  CMAKE_ARGS        -DCMAKE_BUILD_TYPE=Release -DSM=${SM} -DBUILD_PD=ON -DBUILD_ENCODER=${WITH_ENCODER} -DPY_CMD=${PY_CMD} -DON_INFER=${ON_INFER} -DPADDLE_LIB=${PADDLE_LIB} -DWITH_MKL=${WITH_MKL} -DWITH_STATIC_LIB=${WITH_STATIC_LIB} -DBUILD_GPT=OFF
)
# -DBUILD_GPT=${WITH_GPT} 
ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} BINARY_DIR)
ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} SOURCE_DIR)
ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} SOURCE_SUBDIR)

set(FT_INCLUDE_PATH ${SOURCE_DIR}/${SOURCE_SUBDIR})
set(FT_LIB_PATH ${BINARY_DIR}/lib)

include_directories(
  ${FT_INCLUDE_PATH}
)

link_directories(
  ${FT_LIB_PATH}
)

if(ON_INFER AND WITH_GPT AND WITH_SP)
  ExternalProject_Add(
    extern_sentencepiece
    GIT_REPOSITORY    https://github.com/google/sentencepiece.git
    PREFIX            ${THIRD_PATH}
    SOURCE_DIR        ${THIRD_PATH}/source/sentencepiece/
    BINARY_DIR        ${THIRD_PATH}/build/sentencepiece/
    INSTALL_COMMAND   ""
  )
  
  include_directories(
    ${CMAKE_BINARY_DIR}/${THIRD_PATH}/source/sentencepiece/src/
  )

  link_directories(
    ${CMAKE_BINARY_DIR}/${THIRD_PATH}/build/sentencepiece/src/
  )

  add_definitions(-DGPT_ON_SENTENCEPIECE)
endif()

add_subdirectory(faster_transformer)
