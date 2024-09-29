# 海光 DCU 飞桨安装教程

当前 PaddleX 支持海光 Z100 系列芯片。考虑到环境差异性，我们推荐使用**飞桨官方发布的海光 DCU 开发镜像**，该镜像预装有海光 DCU 基础运行环境库（DTK）。

## 1、docker环境准备
拉取镜像，此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包

```
docker pull registry.baidubce.com/device/paddle-dcu:dtk23.10.1-kylinv10-gcc73-py310
```
参考如下命令启动容器

```
docker run -it --name paddle-dcu-dev -v `pwd`:/work \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  registry.baidubce.com/device/paddle-dcu:dtk23.10.1-kylinv10-gcc73-py310 /bin/bash
```

## 2、安装paddle包
在启动的 docker 容器中，下载并安装飞桨官网发布的 wheel 包。**注意**：飞桨框架 DCU 版仅支持海光 C86 架构。

```
# 下载并安装 wheel 包
pip install paddlepaddle-rocm -i https://www.paddlepaddle.org.cn/packages/nightly/dcu
```
验证安装包 安装完成之后，运行如下命令

```
python -c "import paddle; paddle.utils.run_check()"
```
预期得到如下输出结果

```
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```