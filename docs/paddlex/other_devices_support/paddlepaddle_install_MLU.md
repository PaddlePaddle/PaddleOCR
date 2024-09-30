# 寒武纪 MLU 飞桨安装教程

当前 PaddleX 支持寒武纪 MLU370X8 芯片。考虑到环境差异性，我们推荐使用**飞桨官方提供的寒武纪 MLU 开发镜像**完成环境准备。


## 1、docker环境准备
拉取镜像，此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包

```
# 适用于 X86 架构，暂时不提供 Arch64 架构镜像
docker pull registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-gcc84-py310
```
参考如下命令启动容器

```
docker run -it --name paddle-mlu-dev -v $(pwd):/work \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /usr/bin/cnmon:/usr/bin/cnmon \
  registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-gcc84-py310 /bin/bash
```

## 2、安装paddle包
在启动的 docker 容器中，下载并安装飞桨官网发布的 wheel 包。当前提供 Python3.10 的 wheel 安装包。如有其他 Python 版本需求，可以参考[飞桨官方文档](https://www.paddlepaddle.org.cn/install/quick)自行编译安装。

```
# 下载并安装 wheel 包
# 注意需要先安装飞桨 cpu 版本
python -m pip install paddlepaddle==3.0.0.dev20240624 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -m pip install paddle-custom-mlu==3.0.0.dev20240806 -i https://www.paddlepaddle.org.cn/packages/nightly/mlu/
```
验证安装包 安装完成之后，运行如下命令

```
python -c "import paddle; paddle.utils.run_check()"
```
预期得到如下输出结果

```
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 mlu.
PaddlePaddle works well on 16 mlus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```