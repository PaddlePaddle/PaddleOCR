if (${CUDA_VERSION} GREATER_EQUAL 11.0)
  message(STATUS "Add DCUDA11_MODE")
  add_definitions("-DCUDA11_MODE")
endif()

add_definitions(-DNDEBUG)
add_definitions(-DPADDLE_CUDA)
# Default is 1 in standard c++ when using gcc8.2
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

if(ON_INFER)
  add_definitions(-DPADDLE_ON_INFERENCE)

  link_directories(${COMMON_LIB_DIRS})

  set(ft_lib_link
    decoder decoding topk cuda_int8_kernels cuda_kernels online_softmax_beamsearch transformer_kernels attention_kernels encoder nccl_utils nvtx_utils
  )

  if(WITH_GPU)
    add_definitions("-DPADDLE_WITH_CUDA")
  endif()

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
  if(NOT DEFINED DEMO)
    message(FATAL_ERROR "please set DEMO with -DDEMO=demo_name")
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

  if (WIN32)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
    option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)
    if (MSVC_STATIC_CRT)
      if (WITH_MKL)
        set(FLAG_OPENMP "/openmp")
      endif()
      set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
      set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
      set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
      set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
      safe_set_static_flag()
      if (WITH_STATIC_LIB)
        add_definitions(-DSTATIC_LIB)
      endif()
    endif()
  else()
    if(WITH_MKL)
      set(FLAG_OPENMP "-fopenmp")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 ${FLAG_OPENMP}")
  endif()

  if (USE_TENSORRT AND WITH_GPU)
    set(TENSORRT_ROOT "" CACHE STRING "The root directory of TensorRT library")
    if("${TENSORRT_ROOT}" STREQUAL "")
        message(FATAL_ERROR "The TENSORRT_ROOT is empty, you must assign it a value with CMake command. Such as: -DTENSORRT_ROOT=TENSORRT_ROOT_PATH ")
    endif()
    set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
    set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)
  endif()

  if (NOT WIN32)
    if (USE_TENSORRT AND WITH_GPU)
        include_directories("${TENSORRT_INCLUDE_DIR}")
        link_directories("${TENSORRT_LIB_DIR}")
    endif()
  endif(NOT WIN32)

  if(WITH_MKL)
    set(MATH_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mklml")
    include_directories("${MATH_LIB_PATH}/include")
    if(WIN32)
      set(MATH_LIB ${MATH_LIB_PATH}/lib/mklml${CMAKE_STATIC_LIBRARY_SUFFIX}
                  ${MATH_LIB_PATH}/lib/libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})
    else()
      set(MATH_LIB ${MATH_LIB_PATH}/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
                  ${MATH_LIB_PATH}/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    set(MKLDNN_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mkldnn")
    if(EXISTS ${MKLDNN_PATH})
      include_directories("${MKLDNN_PATH}/include")
      if(WIN32)
        set(MKLDNN_LIB ${MKLDNN_PATH}/lib/mkldnn.lib)
      else(WIN32)
        set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
      endif(WIN32)
    endif()
  else()
    set(OPENBLAS_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}openblas")
    include_directories("${OPENBLAS_LIB_PATH}/include/openblas")
    if(WIN32)
      set(MATH_LIB ${OPENBLAS_LIB_PATH}/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
    else()
      set(MATH_LIB ${OPENBLAS_LIB_PATH}/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
  endif()

  if(WITH_STATIC_LIB)
    set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    if(WIN32)
      set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
    else()
      set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
  endif()

  if (NOT WIN32)
    set(EXTERNAL_LIB "-lrt -ldl -lpthread")
    set(DEPS ${DEPS}
        ${MATH_LIB} ${MKLDNN_LIB}
        glog gflags protobuf  xxhash
        ${EXTERNAL_LIB})
  else()
    set(DEPS ${DEPS}
        ${MATH_LIB} ${MKLDNN_LIB}
        glog gflags_static libprotobuf xxhash ${EXTERNAL_LIB})
    set(DEPS ${DEPS} shlwapi.lib)
  endif(NOT WIN32)

  cuda_add_library(decoding_infer_op ${decoding_op_files} SHARED)
  add_dependencies(decoding_infer_op extern_${THIRD_PARTY_NAME} boost)

  string(REPLACE "/" ";" DEMO_PATH ${DEMO})

  list(LENGTH DEMO_PATH PATH_LEN)
  MATH(EXPR PATH_LEN "${PATH_LEN}-1")
  list(GET DEMO_PATH ${PATH_LEN} DEMO_NAME)

  string(REPLACE "." ";" DEMO_NAME ${DEMO_NAME})
  list(GET DEMO_NAME 0 DEMO_NAME)
  add_executable(${DEMO_NAME} ${DEMO})
  set(DEPS decoding_infer_op ${ft_lib_link} boost ${DEPS} cublas cudart)

  if(WITH_GPT AND WITH_SP)
    set(DEPS ${DEPS} sentencepiece)
    add_dependencies(decoding_infer_op extern_sentencepiece)
  endif()

  if(WIN32)
    if(USE_TENSORRT)
      add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E copy ${TENSORRT_LIB_DIR}/nvinfer${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
              COMMAND ${CMAKE_COMMAND} -E copy ${TENSORRT_LIB_DIR}/nvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
      )
    endif()
    if(WITH_MKL)
      add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${MATH_LIB_PATH}/lib/mklml.dll ${CMAKE_BINARY_DIR}/Release
            COMMAND ${CMAKE_COMMAND} -E copy ${MATH_LIB_PATH}/lib/libiomp5md.dll ${CMAKE_BINARY_DIR}/Release
            COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_PATH}/lib/mkldnn.dll  ${CMAKE_BINARY_DIR}/Release
      )
    else()
      add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${OPENBLAS_LIB_PATH}/lib/openblas.dll ${CMAKE_BINARY_DIR}/Release
      )
    endif()
    if(NOT WITH_STATIC_LIB)
        add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_LIB}/paddle/lib/paddle_fluid.dll" ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
        )
    endif()
  endif()

  target_link_libraries(${DEMO_NAME} ${DEPS})

else(ON_INFER)
  if(NOT PY_CMD)
    set(PYTHON_PATH "python" CACHE STRING "Python path")
  else()
    set(PYTHON_PATH ${PY_CMD} CACHE STRING "Python path")
  endif()

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

  include_directories(
    ${COMMON_HEADER_DIRS}
  )

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

  link_directories(
    ${COMMON_LIB_DIRS}
  )

  include_directories(${PY_INCLUDE_DIR})
  include_directories(${PY_INCLUDE_DIR}/third_party)

  if(EXISTS ${PY_LIB_DIR}/libpaddle_custom_op.so)
    set(lib_link
      -lpaddle_custom_op
    )
  endif()

  if(EXISTS ${PY_LIB_DIR}/../fluid/core_avx.so)
    set(lib_link
      -l:core_avx.so
    )
    link_directories(
      ${PY_LIB_DIR}/../fluid/
    )
  else()
    set(lib_link
      -l:core_noavx.so
    )
    link_directories(
      ${PY_LIB_DIR}/../fluid/
    )
  endif()

  set(ft_lib_link
    -ldecoder -ldecoding -ltopk -lcuda_int8_kernels -lcuda_kernels -lonline_softmax_beamsearch -ltransformer_kernels -lattention_kernels -lencoder -lnccl_utils -lnvtx_utils
  )

  add_definitions(-DPADDLE_WITH_CUDA)
  add_definitions(-DEIGEN_USE_GPU)
  add_definitions(-DPADDLE_USE_DSO)
  if (WITH_MKL)
    add_definitions(-DPADDLE_WITH_MKLDNN)
  endif()

  add_library(decoding_op SHARED ${decoding_op_files})
  add_dependencies(decoding_op extern_${THIRD_PARTY_NAME} boost)
  target_link_libraries(decoding_op PRIVATE -lcublas -lcudart ${lib_link} ${ft_lib_link})
endif()
