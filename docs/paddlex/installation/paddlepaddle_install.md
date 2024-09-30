# 飞桨PaddlePaddle本地安装教程



安装飞桨 PaddlePaddle 时，支持通过 Docker 安装和通过 pip 安装。

## 基于 Docker 安装飞桨
**若您通过 Docker 安装**，请参考下述命令，使用飞桨官方 Docker 镜像，创建一个名为 `paddlex` 的容器，并将当前工作目录映射到容器内的 `/paddle` 目录：

```bash
# 对于 gpu 用户
# CUDA11.8 用户
nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it registry.baidubce.com/paddlepaddle/paddle:3.0.0b1-gpu-cuda11.8-cudnn8.6-trt8.5 /bin/bash

# CUDA12.3 用户
nvidia-docker run --name paddlex -v $PWD:/paddle  --shm-size=8G --network=host -it registry.baidubce.com/paddlepaddle/paddle:3.0.0b1-gpu-cuda12.3-cudnn9.0-trt8.6 /bin/bash
```
注：更多飞桨官方 docker 镜像请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html)。若您是 CUDA11.8 用户，请确保您的 Docker版本 >= 19.03；若您是 CUDA12.3 用户，请确保您的 Docker版本 >= 20.10。

## 基于 pip 安装飞桨
**若您通过 pip 安装**，请参考下述命令，用 pip 在当前环境中安装飞桨 PaddlePaddle：

```bash
# cpu
python -m pip install paddlepaddle

# gpu，该命令仅适用于 CUDA 版本为 11.8 的机器环境
 python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu，该命令仅适用于 CUDA 版本为 12.3 的机器环境
 python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```
> ❗ **注**：更多飞桨 Wheel 版本请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

**关于其他硬件安装飞桨，请参考**[多硬件安装飞桨](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/tutorials/INSTALL_OTHER_DEVICES.md)**。**

安装完成后，使用以下命令可以验证 PaddlePaddle 是否安装成功：

```bash
python -c "import paddle; print(paddle.__version__)"
```
如果已安装成功，将输出以下内容：

```bash
3.0.0-beta1
```