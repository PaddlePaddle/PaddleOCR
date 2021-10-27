
## 环境配置

本教程适用于PTDN目录下基础功能测试的运行环境搭建。

推荐环境：
- CUDA 10.1
- CUDNN 7.6
- TensorRT 6.1.0.5 / 7.1


推荐docker镜像安装，按照如下命令创建镜像，当前目录映射到镜像中的`/paddle`目录下
```
nvidia-docker run --name paddle -it -v $PWD:/paddle paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82 /bin/bash
cd /paddle

# 安装带TRT的paddle
pip3.7 install https://paddle-wheel.bj.bcebos.com/with-trt/2.1.3/linux-gpu-cuda10.1-cudnn7-mkl-gcc8.2-trt6-avx/paddlepaddle_gpu-2.1.3.post101-cp37-cp37m-linux_x86_64.whl

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
或者问题是下载的TensorRT版本和当前paddle中编译的TRT版本不匹配，需要下载版本相符的TRT。
