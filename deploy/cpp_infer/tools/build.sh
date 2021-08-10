set -o errexit

if [ $# != 1 ] ; then
echo "USAGE: $0 MODE (one of ['det', 'rec', 'system'])"
echo " e.g.: $0 system"
exit 1;
fi

# MODE be one of ['det', 'rec', 'system']
MODE=$1
cp CMakeLists_$MODE.txt CMakeLists.txt


OPENCV_DIR=/paddle/git/new/PaddleOCR/deploy/cpp_infer/opencv-3.4.7/opencv3/
LIB_DIR=/paddle/git/new/PaddleOCR/deploy/cpp_infer/paddle_inference/
CUDA_LIB_DIR=/usr/local/cuda/lib64/
CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/


BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF \
    -DWITH_TENSORRT=OFF \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \
    -DTENSORRT_DIR=${TENSORRT_DIR} \

make -j
