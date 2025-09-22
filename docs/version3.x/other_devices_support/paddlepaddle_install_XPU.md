---
comments: true
---

# 昆仑 XPU 飞桨安装教程

当前 PaddleOCR 支持昆仑 R200/R300 等芯片。考虑到环境差异性，我们推荐使用<b>飞桨官方发布的昆仑 XPU 开发镜像</b>，该镜像预装有昆仑基础运行环境库（XRE）。

## 1、docker环境准备
拉取镜像，此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包

```
docker pull registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310 # X86 架构
docker pull registry.baidubce.com/device/paddle-xpu:kylinv10-aarch64-gcc82-py310 # ARM 架构
```
参考如下命令启动容器

```
docker run -it --name=xxx -m 81920M --memory-swap=81920M \
    --shm-size=128G --privileged --net=host \
    -v $(pwd):/workspace -w /workspace \
    registry.baidubce.com/device/paddle-xpu:$(uname -m)-py310 bash
```
## 2、安装paddle包
当前提供 Python3.10 的 wheel 安装包。如有其他 Python 版本需求，可以参考[飞桨官方文档](https://www.paddlepaddle.org.cn/install/quick)自行编译安装。

安装 Python3.10 的 wheel 安装包

```
pip install https://paddle-whl.bj.bcebos.com/paddlex/xpu/paddlepaddle_xpu-2.6.1-cp310-cp310-linux_x86_64.whl # X86 架构
pip install https://paddle-whl.bj.bcebos.com/paddlex/xpu/paddlepaddle_xpu-2.6.1-cp310-cp310-linux_aarch64.whl # ARM 架构
```
验证安装包 安装完成之后，运行如下命令

```
python -c "import paddle; paddle.utils.run_check()"
```
预期得到如下输出结果

```
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
