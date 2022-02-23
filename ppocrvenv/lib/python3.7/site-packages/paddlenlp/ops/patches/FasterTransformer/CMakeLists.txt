# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(FasterTransformer LANGUAGES CXX CUDA)

find_package(CUDA 10.1 REQUIRED)

option(BUILD_PD "Build in PaddlePaddle mode" ON)
option(BUILD_GPT "Build project with gpt"    ON)
option(BUILD_ENCODER "Build project with encoder"    ON)

if(BUILD_ENCODER)
  add_definitions(-DBUILD_ENCODER)
endif()

if(BUILD_GPT)
  message(STATUS "Add DBUILD_GPT, requires MPI and NCCL")
  add_definitions("-DBUILD_GPT")
  set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
  find_package(MPI REQUIRED)
  find_package(NCCL REQUIRED)
  #if(${NCCL_VERSION} LESS 2.7)
  #  message(FATAL_ERROR "NCCL_VERSION ${NCCL_VERSION} is less than 2.7")
  #endif()
  set(CMAKE_MODULE_PATH "") # prevent the bugs for pytorch building
endif()

set(CXX_STD "14" CACHE STRING "C++ standard")

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

if (${CUDA_VERSION} GREATER_EQUAL 11.0)
  message(STATUS "Add DCUDA11_MODE")
  add_definitions("-DCUDA11_MODE")
endif()

# profiling
option(USE_NVTX "Whether or not to use nvtx" OFF)
if(USE_NVTX)
  message(STATUS "NVTX is enabled.")
  add_definitions("-DUSE_NVTX")
endif()

# setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall -ldl")

# if (SM STREQUAL 80 OR
#     SM STREQUAL 86 OR
#     SM STREQUAL 70 OR
#     SM STREQUAL 75 OR
#     SM STREQUAL 61 OR
#     SM STREQUAL 60)
# #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM},code=\\\"sm_${SM},compute_${SM}\\\" -rdc=true")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM},code=\\\"sm_${SM},compute_${SM}\\\"")
#   if (SM STREQUAL 70 OR SM STREQUAL 75 OR SM STREQUAL 80 OR SM STREQUAL 86)
#     set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
#     set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
#   endif()
# message("-- Assign GPU architecture (sm=${SM})")

# else()
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
#                       -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
#                       -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
#                       ")
# #                      -rdc=true")
# set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
# set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
# message("-- Assign GPU architecture (sm=70,75)")
# endif()

set(SM_SETS 52 60 61 70 75 80)
set(USING_WMMA False)
set(FIND_SM False)

foreach(SM_NUM IN LISTS SM_SETS)
  string(FIND "${SM}" "${SM_NUM}" SM_POS)
  if(SM_POS GREATER -1)
    if(FIND_SM STREQUAL False)
      set(ENV{TORCH_CUDA_ARCH_LIST} "")
    endif()
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
  #                      -rdc=true")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")

  set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
  message("-- Assign GPU architecture (sm=70,75,80)")
endif()

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall  --ptxas-options=-v --resource-usage")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD}")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
# set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 --ptxas-options=--verbose")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
)

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

if(NOT PY_CMD)
  set(PYTHON_PATH "python" CACHE STRING "Python path")
else()
  set(PYTHON_PATH ${PY_CMD} CACHE STRING "Python path")
endif()

if(BUILD_PD)
  add_definitions(-DPADDLE_WITH_CUDA)

  if(ON_INFER)
    add_definitions(-DPADDLE_ON_INFERENCE)

    link_directories(${COMMON_LIB_DIRS})

    if(NOT WITH_STATIC_LIB)
      add_definitions("-DPADDLE_WITH_SHARED_LIB")
    else()
      # PD_INFER_DECL is mainly used to set the dllimport/dllexport attribute in dynamic library mode.
      # Set it to empty in static library mode to avoid compilation issues.
      add_definitions("/DPD_INFER_DECL=")
    endif()

    macro(safe_set_static_flag)
        foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
          if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
          endif(${flag_var} MATCHES "/MD")
        endforeach(flag_var)
    endmacro()

    if(NOT DEFINED PADDLE_LIB)
      message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
    endif()

    include_directories("${PADDLE_LIB}/")
    set(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_LIB}/third_party/install/")
    include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
    include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
    include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
    include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")

    link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
    link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/lib")
    link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/lib")
    link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
    link_directories("${PADDLE_LIB}/paddle/lib")

    if(WITH_MKL)
      set(FLAG_OPENMP "-fopenmp")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 ${FLAG_OPENMP}")

    if (USE_TENSORRT AND WITH_GPU)
      set(TENSORRT_ROOT "" CACHE STRING "The root directory of TensorRT library")
      if("${TENSORRT_ROOT}" STREQUAL "")
          message(FATAL_ERROR "The TENSORRT_ROOT is empty, you must assign it a value with CMake command. Such as: -DTENSORRT_ROOT=TENSORRT_ROOT_PATH ")
      endif()
      set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
      set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)
    endif()

    if (USE_TENSORRT AND WITH_GPU)
        include_directories("${TENSORRT_INCLUDE_DIR}")
        link_directories("${TENSORRT_LIB_DIR}")
    endif()

    if(WITH_MKL)
      set(MATH_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mklml")
      include_directories("${MATH_LIB_PATH}/include")
      set(MKLDNN_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mkldnn")
      if(EXISTS ${MKLDNN_PATH})
        include_directories("${MKLDNN_PATH}/include")
        set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
      endif()
    else()
      set(OPENBLAS_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}openblas")
      include_directories("${OPENBLAS_LIB_PATH}/include/openblas")
    endif()

  else()
    execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import paddle; print(paddle.sysconfig.get_include())"
                    RESULT_VARIABLE _INC_PYTHON_SUCCESS
                    OUTPUT_VARIABLE _INC_PYTHON_VALUES)
    if (NOT _INC_PYTHON_SUCCESS MATCHES 0)
        message(FATAL_ERROR "Python config Error.")
    endif()
    string(REGEX REPLACE ";" "\\\\;" _INC_PYTHON_VALUES ${_INC_PYTHON_VALUES})
    string(REGEX REPLACE "\n" ";" _INC_PYTHON_VALUES ${_INC_PYTHON_VALUES})
    list(GET _INC_PYTHON_VALUES 0 PY_INCLUDE_DIR)

    list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})
    list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR}/third_party)

    execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import paddle; print(paddle.sysconfig.get_lib())"
                    RESULT_VARIABLE _LIB_PYTHON_SUCCESS
                    OUTPUT_VARIABLE _LIB_PYTHON_VALUES)
    if (NOT _LIB_PYTHON_SUCCESS MATCHES 0)
        message(FATAL_ERROR "Python config Error.")
    endif()
    string(REGEX REPLACE ";" "\\\\;" _LIB_PYTHON_VALUES ${_LIB_PYTHON_VALUES})
    string(REGEX REPLACE "\n" ";" _LIB_PYTHON_VALUES ${_LIB_PYTHON_VALUES})
    list(GET _LIB_PYTHON_VALUES 0 PY_LIB_DIR)
    list(APPEND COMMON_LIB_DIRS ${PY_LIB_DIR})

    include_directories(${PY_INCLUDE_DIR})
    include_directories(${PY_INCLUDE_DIR}\third_party)

  endif()
endif()

list(APPEND COMMON_HEADER_DIRS ${MPI_INCLUDE_PATH})

include_directories(
  ${COMMON_HEADER_DIRS}
)

list(APPEND COMMON_LIB_DIRS /usr/local/mpi/lib)

link_directories(
  ${COMMON_LIB_DIRS}
)

add_subdirectory(fastertransformer)
add_subdirectory(tools)
add_subdirectory(sample)

########################################

if(BUILD_GPT)
# Following feature requires cmake 3.15
# TODO Remove this part or modify such that we can run it under cmake 3.10
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
add_library(transformer-static STATIC
  $<TARGET_OBJECTS:encoder>
  $<TARGET_OBJECTS:cuda_kernels>
  $<TARGET_OBJECTS:transformer_kernels>
  $<TARGET_OBJECTS:nvtx_utils>
  $<TARGET_OBJECTS:cuda_int8_kernels>
  $<TARGET_OBJECTS:attention_kernels>
  $<TARGET_OBJECTS:trt_fused_multi_head_attention>
  $<TARGET_OBJECTS:encoder_gemm_func>
  $<TARGET_OBJECTS:encoder_igemm_func>
  $<TARGET_OBJECTS:decoder>
  $<TARGET_OBJECTS:decoding>
  $<TARGET_OBJECTS:topk>
  $<TARGET_OBJECTS:online_softmax_beamsearch>
  $<TARGET_OBJECTS:nccl_utils>)
set_property(TARGET transformer-static PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET transformer-static PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(transformer-static PUBLIC -lcublas -lcudart -lcurand -lnccl -lmpi nvtx_utils)

add_library(transformer-shared SHARED
  $<TARGET_OBJECTS:encoder>
  $<TARGET_OBJECTS:cuda_kernels>
  $<TARGET_OBJECTS:transformer_kernels>
  $<TARGET_OBJECTS:nvtx_utils>
  $<TARGET_OBJECTS:cuda_int8_kernels>
  $<TARGET_OBJECTS:attention_kernels>
  $<TARGET_OBJECTS:trt_fused_multi_head_attention>
  $<TARGET_OBJECTS:encoder_gemm_func>
  $<TARGET_OBJECTS:encoder_igemm_func>
  $<TARGET_OBJECTS:decoder>
  $<TARGET_OBJECTS:decoding>
  $<TARGET_OBJECTS:topk>
  $<TARGET_OBJECTS:online_softmax_beamsearch>
  $<TARGET_OBJECTS:nccl_utils>
  $<TARGET_OBJECTS:gpt_triton_backend>)
## add_library(transformer-shared SHARED  $<TARGET_OBJECTS:encoder>)
set_target_properties(transformer-shared PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(transformer-shared PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(transformer-shared PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(transformer-shared PUBLIC ${NCCL_LIBRARIES} ${MPI_LIBRARIES} -lcublas -lcublasLt -lcudart -lcurand )

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/FasterTransformer)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/FasterTransformerConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/FasterTransformerConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/FasterTransformerConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  TARGETS
    transformer-shared
  EXPORT
    transformer-shared-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/fastertransformer
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/fastertransformer
)

install(
  EXPORT
    transformer-shared-targets
  FILE
    FasterTransformerTargets.cmake
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

file(GLOB_RECURSE HEADER_FILES "*.h" "*.hpp" "*.cuh")
foreach ( file ${HEADER_FILES} )
    file( RELATIVE_PATH rfile ${CMAKE_CURRENT_SOURCE_DIR} ${file} )
    get_filename_component( dir ${rfile} DIRECTORY )
    install( FILES ${file} DESTINATION  ${CMAKE_INSTALL_PREFIX}/backends/fastertransformer/include/${dir} )
endforeach()


################################################################################
add_executable(gpt sample/cpp/gpt_sample.cc )
target_link_libraries(gpt PUBLIC -lcublas -lcublasLt -lcudart -lcurand -lnccl -lmpi transformer-static)
# target_link_libraries(gpt PUBLIC -lcublas -lcublasLt -lcudart -lcurand -lnccl -lmpi decoder decoding)

export(
  EXPORT
    transformer-shared-targets
  FILE
    ${CMAKE_CURRENT_BINARY_DIR}/FasterTransformerTargets.cmake
  NAMESPACE
    TritonCore::
)

export(PACKAGE FasterTransformer)

endif() # BUILD_GPT
