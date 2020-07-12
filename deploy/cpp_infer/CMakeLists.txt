project(ocr_system CXX C)
option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       ON)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)
option(USE_TENSORRT "Compile demo with TensorRT."   OFF)


macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fpermissive")
set(CMAKE_STATIC_LIBRARY_PREFIX "")
message("flags" ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()
if(NOT DEFINED DEMO_NAME)
  message(FATAL_ERROR "please set DEMO_NAME with -DDEMO_NAME=demo_name")
endif()

# user ze
# find_package(OpenCV)


set(OPENCV_DIR "/paddle/libs/opencv-3.4.7/opencv3")
find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("${PADDLE_LIB}/paddle/include")
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")

include_directories("${CMAKE_SOURCE_DIR}/")

if (USE_TENSORRT AND WITH_GPU)
      include_directories("${TENSORRT_ROOT}/include")
      link_directories("${TENSORRT_ROOT}/lib")
endif()

link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")

link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/paddle/lib")


add_executable(${DEMO_NAME} src/main.cpp src/ocr_det.cpp src/ocr_rec.cpp src/preprocess_op.cpp src/clipper.cpp src/postprocess_op.cpp )

if(WITH_MKL)
  include_directories("${PADDLE_LIB}/third_party/install/mklml/include")
  set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
               ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(MKLDNN_PATH "${PADDLE_LIB}/third_party/install/mkldnn")
  if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
  endif()
else()
  set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

# Note: libpaddle_inference_api.so/a must put before libpaddle_fluid.so/a
if(WITH_STATIC_LIB)
  set(DEPS
      ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(DEPS
      ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

# user ze
# set(EXTERNAL_LIB "-lrt -ldl -lpthread -lm -lopencv_world")
# gry
set(EXTERNAL_LIB "-lrt -ldl -lpthread -lm")

set(DEPS ${DEPS}
    ${MATH_LIB} ${MKLDNN_LIB}
    glog gflags protobuf z xxhash
    ${EXTERNAL_LIB} ${OpenCV_LIBS})

if(WITH_GPU)
  if (USE_TENSORRT)
    set(DEPS ${DEPS}
        ${TENSORRT_ROOT}/lib/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DEPS ${DEPS}
        ${TENSORRT_ROOT}/lib/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
  set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX} )
  set(DEPS ${DEPS} ${CUDA_LIB}/libcublas${CMAKE_SHARED_LIBRARY_SUFFIX} )
  set(DEPS ${DEPS} ${CUDNN_LIB}/libcudnn${CMAKE_SHARED_LIBRARY_SUFFIX} )
endif()

target_link_libraries(${DEMO_NAME} ${DEPS})
