---
comments: true
typora-copy-images-to: images
---

# 多硬件安装飞桨

本文档主要针对昇腾 NPU 硬件平台，介绍如何安装飞桨。

## 1. 昇腾 NPU 飞桨安装

### 1.1 环境准备

当前 PaddleOCR 支持昇腾 910B 芯片，昇腾驱动版本为 23.0.3。考虑到环境差异性，我们推荐使用飞桨官方提供的标准镜像完成环境准备。

#### 拉取镜像

此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包，镜像中已经默认安装了昇腾算子库 CANN-8.0.RC1。

```bash linenums="1"
# 适用于 X86 架构，暂时不提供 Arch64 架构镜像
docker pull registry.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-x86_64-gcc84-py39
```

#### 启动容器

ASCEND_RT_VISIBLE_DEVICES 指定可见的 NPU 卡号

```bash linenums="1"
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-x86_64-gcc84-py39 /bin/bash
```

### 1.2 安装 paddle 包

当前提供 Python3.9 的 wheel 安装包。如有其他 Python 版本需求，可以参考[飞桨官方文档](https://www.paddlepaddle.org.cn/install/quick)自行编译安装。

#### 1. 下载安装 Python3.9 的 wheel 安装包

```bash linenums="1"
# 注意需要先安装飞桨 cpu 版本
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddle-device/npu/paddlepaddle-0.0.0-cp39-cp39-linux_x86_64.whl
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddle-device/npu/paddle_custom_npu-0.0.0-cp39-cp39-linux_x86_64.whl
```

#### 2. 验证安装包

安装完成之后，运行如下命令。

```bash linenums="1"
python -c "import paddle; paddle.utils.run_check()"
```

预期得到如下输出结果

```bash linenums="1"
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 npu.
PaddlePaddle works well on 8 npus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
