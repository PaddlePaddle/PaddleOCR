## 1. 环境准备

本教程适用于test_tipc目录下基础功能测试的运行环境搭建。

推荐环境：
- CUDA 10.1/10.2
- CUDNN 7.6/cudnn8.1
- TensorRT 6.1.0.5 / 7.1 / 7.2

环境配置可以选择docker镜像安装，或者在本地环境Python搭建环境。推荐使用docker镜像安装，避免不必要的环境配置。

## 2. Docker 镜像安装

推荐docker镜像安装，按照如下命令创建镜像，当前目录映射到镜像中的`/paddle`目录下
```
nvidia-docker run --name paddle -it -v $PWD:/paddle paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82 /bin/bash
cd /paddle

# 安装带TRT的paddle
pip3.7 install https://paddle-wheel.bj.bcebos.com/with-trt/2.1.3/linux-gpu-cuda10.1-cudnn7-mkl-gcc8.2-trt6-avx/paddlepaddle_gpu-2.1.3.post101-cp37-cp37m-linux_x86_64.whl
```

## 3 Python 环境构建

非docker环境下，环境配置比较灵活，推荐环境组合配置：
- CUDA10.1 + CUDNN7.6 + TensorRT 6
- CUDA10.2 + CUDNN8.1 + TensorRT 7
- CUDA11.1 + CUDNN8.1 + TensorRT 7

下面以 CUDA10.2 + CUDNN8.1 + TensorRT 7 配置为例，介绍环境配置的流程。

### 3.1 安装CUDNN

如果当前环境满足CUDNN版本的要求，可以跳过此步骤。

以CUDNN8.1 安装安装为例，安装步骤如下，首先下载CUDNN，从[Nvidia官网](https://developer.nvidia.com/rdp/cudnn-archive)下载CUDNN8.1版本，下载符合当前系统版本的三个deb文件，分别是：
- cuDNN Runtime Library ，如：libcudnn8_8.1.0.77-1+cuda10.2_amd64.deb
- cuDNN Developer Library ，如：libcudnn8-dev_8.1.0.77-1+cuda10.2_amd64.deb
- cuDNN Code Samples，如：libcudnn8-samples_8.1.0.77-1+cuda10.2_amd64.deb

deb安装可以参考[官方文档](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb)，安装方式如下
```
# x.x.x表示下载的版本号
# $HOME为工作目录
sudo dpkg -i libcudnn8_x.x.x-1+cudax.x_arm64.deb
sudo dpkg -i libcudnn8-dev_8.x.x.x-1+cudax.x_arm64.deb
sudo dpkg -i libcudnn8-samples_8.x.x.x-1+cudax.x_arm64.deb

# 验证是否正确安装
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd  $HOME/cudnn_samples_v8/mnistCUDNN

# 编译
make clean && make
./mnistCUDNN
```
如果运行mnistCUDNN完后提示运行成功，则表示安装成功。如果运行后出现freeimage相关的报错，需要按照提示安装freeimage库:
```
sudo apt-get install libfreeimage-dev
sudo apt-get install libfreeimage
```

### 3.2 安装TensorRT

首先，从[Nvidia官网TensorRT板块](https://developer.nvidia.com/tensorrt-getting-started)下载TensorRT，这里选择7.1.3.4版本的TensorRT，注意选择适合自己系统版本和CUDA版本的TensorRT，另外建议下载TAR package的安装包。

以Ubuntu16.04+CUDA10.2为例，下载并解压后可以参考[官方文档](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/install-guide/index.html#installing-tar)的安装步骤，按照如下步骤安装:
```
# 以下安装命令中 '${version}' 为下载的TensorRT版本，如7.1.3.4
# 设置环境变量，<TensorRT-${version}/lib> 为解压后的TensorRT的lib目录
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>

# 安装TensorRT
cd TensorRT-${version}/python
pip3.7 install tensorrt-*-cp3x-none-linux_x86_64.whl

# 安装graphsurgeon
cd TensorRT-${version}/graphsurgeon
```


### 3.3 安装PaddlePaddle

下载支持TensorRT版本的Paddle安装包，注意安装包的TensorRT版本需要与本地TensorRT一致，下载[链接](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#python)
选择下载 linux-cuda10.2-trt7-gcc8.2 Python3.7版本的Paddle：
```
# 从下载链接中可以看到是paddle2.1.1-cuda10.2-cudnn8.1版本
wget  https://paddle-wheel.bj.bcebos.com/with-trt/2.1.1-gpu-cuda10.2-cudnn8.1-mkl-gcc8.2/paddlepaddle_gpu-2.1.1-cp37-cp37m-linux_x86_64.whl
pip3.7 install -U paddlepaddle_gpu-2.1.1-cp37-cp37m-linux_x86_64.whl
```

## 4. 安装PaddleOCR依赖
```
# 安装AutoLog
git clone https://github.com/LDOUBLEV/AutoLog
cd AutoLog
pip3.7 install -r requirements.txt
python3.7 setup.py bdist_wheel
pip3.7 install ./dist/auto_log-1.0.0-py3-none-any.whl

# 下载OCR代码
cd ../
git clone https://github.com/PaddlePaddle/PaddleOCR

```

安装PaddleOCR依赖：
```
cd PaddleOCR
pip3.7 install -r requirements.txt
```

## FAQ :
Q. You are using Paddle compiled with TensorRT, but TensorRT dynamic library is not found. Ignore this if TensorRT is not needed.

A. 问题一般是当前安装paddle版本带TRT，但是本地环境找不到TensorRT的预测库，需要下载TensorRT库，解压后设置环境变量LD_LIBRARY_PATH;
如：
```
export LD_LIBRARY_PATH=/usr/local/python3.7.0/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/paddle/package/TensorRT-6.0.1.5/lib
```
或者问题是下载的TensorRT版本和当前paddle中编译的TRT版本不匹配，需要下载版本相符的TensorRT重新安装。
