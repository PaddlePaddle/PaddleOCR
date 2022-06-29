#使用镜像：
#registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82

#编译Serving Server：

#client和app可以直接使用release版本

#server因为加入了自定义OP，需要重新编译

apt-get update
apt install -y libcurl4-openssl-dev libbz2-dev
wget https://paddle-serving.bj.bcebos.com/others/centos_ssl.tar && tar xf centos_ssl.tar && rm -rf centos_ssl.tar && mv libcrypto.so.1.0.2k /usr/lib/libcrypto.so.1.0.2k && mv libssl.so.1.0.2k /usr/lib/libssl.so.1.0.2k && ln -sf /usr/lib/libcrypto.so.1.0.2k /usr/lib/libcrypto.so.10 && ln -sf /usr/lib/libssl.so.1.0.2k /usr/lib/libssl.so.10 && ln -sf /usr/lib/libcrypto.so.10 /usr/lib/libcrypto.so && ln -sf /usr/lib/libssl.so.10 /usr/lib/libssl.so

# 安装go依赖
rm -rf /usr/local/go
wget -qO- https://paddle-ci.cdn.bcebos.com/go1.17.2.linux-amd64.tar.gz | tar -xz -C /usr/local
export GOROOT=/usr/local/go
export GOPATH=/root/gopath
export PATH=$PATH:$GOPATH/bin:$GOROOT/bin
go env -w GO111MODULE=on
go env -w GOPROXY=https://goproxy.cn,direct
go install github.com/grpc-ecosystem/grpc-gateway/protoc-gen-grpc-gateway@v1.15.2
go install github.com/grpc-ecosystem/grpc-gateway/protoc-gen-swagger@v1.15.2
go install github.com/golang/protobuf/protoc-gen-go@v1.4.3
go install google.golang.org/grpc@v1.33.0
go env -w GO111MODULE=auto

# 下载opencv库
wget https://paddle-qa.bj.bcebos.com/PaddleServing/opencv3.tar.gz && tar -xvf opencv3.tar.gz && rm -rf opencv3.tar.gz
export OPENCV_DIR=$PWD/opencv3

# clone Serving
git clone https://github.com/PaddlePaddle/Serving.git -b develop --depth=1
cd Serving
export Serving_repo_path=$PWD
git submodule update --init --recursive
python -m pip install -r python/requirements.txt


export PYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
export PYTHON_LIBRARIES=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
export PYTHON_EXECUTABLE=`which python`

export CUDA_PATH='/usr/local/cuda'
export CUDNN_LIBRARY='/usr/local/cuda/lib64/'
export CUDA_CUDART_LIBRARY='/usr/local/cuda/lib64/'
export TENSORRT_LIBRARY_PATH='/usr/local/TensorRT6-cuda10.1-cudnn7/targets/x86_64-linux-gnu/'

# cp 自定义OP代码
cp -rf ../deploy/pdserving/general_detection_op.cpp ${Serving_repo_path}/core/general-server/op

# 编译Server, export SERVING_BIN
mkdir server-build-gpu-opencv && cd server-build-gpu-opencv
cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
            -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
            -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
            -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH} \
            -DCUDNN_LIBRARY=${CUDNN_LIBRARY} \
            -DCUDA_CUDART_LIBRARY=${CUDA_CUDART_LIBRARY} \
            -DTENSORRT_ROOT=${TENSORRT_LIBRARY_PATH} \
            -DOPENCV_DIR=${OPENCV_DIR} \
            -DWITH_OPENCV=ON \
            -DSERVER=ON \
            -DWITH_GPU=ON ..
make -j32

python -m pip install python/dist/paddle*
export SERVING_BIN=$PWD/core/general-server/serving
cd ../../
