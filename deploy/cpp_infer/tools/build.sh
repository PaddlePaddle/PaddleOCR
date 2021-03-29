OPENCV_DIR=/paddle/Paddle/opencv-3.4.7/opencv3
#LIB_DIR=/paddle/deploy/cuda10.1_cudnn7.6_avx_mkl_trt6_gcc82_paddle_inference/
#LID_DIR=/padle/deploy/2.0.0-rc1-gpu-cuda10.2-cudnn8-avx-mkl-trt7_inference
#LIB_DIR=/paddle/deploy/2.0.0-rc1-gpu-cuda11-cudnn8-avx-mkl-trt7_inference
LIB_DIR=/paddle/Paddle/inference/paddle_inference
#PADDLE_LIB=/paddle/deploy/paddle_inference
CUDA_LIB_DIR=/usr/local/cuda/lib64
CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu
TENSORRT_DIR=/paddle/Paddle/package/TensorRT/TensorRT-6.0.1.8/

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DWITH_GPU=ON \
    -DWITH_STATIC_LIB=OFF \
    -DWITH_TENSORRT=OFF \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \
    -DTENSORRT_DIR=${TENSORRT_DIR} \

make -j
