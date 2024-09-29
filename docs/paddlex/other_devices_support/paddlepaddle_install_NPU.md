# 昇腾 NPU 飞桨安装教程

当前 PaddleX 支持昇腾 910B 芯片（更多型号还在支持中，如果您有其他型号的相关需求，请提交issue告知我们），昇腾驱动版本为 23.0.3。考虑到环境差异性，我们推荐使用**飞桨官方提供的昇腾开发镜像**完成环境准备。

## 1、docker环境准备
* 拉取镜像，此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包，镜像中已经默认安装了昇腾算子库 CANN-8.0.T13。
```
# 适用于 X86 架构
docker pull registry.baidubce.com/device/paddle-npu:cann80T13-ubuntu20-x86_64-gcc84-py39
# 适用于 Aarch64 架构
docker pull registry.baidubce.com/device/paddle-npu:cann80T13-ubuntu20-aarch64-gcc84-py39
```
* 参考如下命令启动容器，ASCEND_RT_VISIBLE_DEVICES 指定可见的 NPU 卡号
```
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann80T13-ubuntu20-$(uname -m)-gcc84-py39 /bin/bash
```
## 2、安装paddle包
当前提供 Python3.9 的 wheel 安装包。如有其他 Python 版本需求，可以参考[飞桨官方文档](https://www.paddlepaddle.org.cn/install/quick)自行编译安装。

* 下载安装 Python3.9 的 wheel 安装包
```
# 注意需要先安装飞桨 cpu 版本
python3.9 -m pip install paddlepaddle==3.0.0.dev20240520 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python3.9 -m pip install paddle_custom_npu==3.0.0.dev20240719 -i https://www.paddlepaddle.org.cn/packages/nightly/npu/
```
* 验证安装包安装完成之后，运行如下命令
```
python -c "import paddle; paddle.utils.run_check()"
```
预期得到如下输出结果

```
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 npu.
PaddlePaddle works well on 8 npus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```