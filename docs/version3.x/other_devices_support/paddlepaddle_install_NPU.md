---
comments: true
---

# 昇腾 NPU 飞桨安装教程

当前 PaddleOCR 支持昇腾 910B 芯片（更多型号还在支持中，如果您有其他型号的相关需求，请提交issue告知我们），昇腾驱动版本为 23.0.3。考虑到环境差异性，我们推荐使用<b>飞桨官方提供的昇腾开发镜像</b>完成环境准备。

## 1、docker环境准备
* 拉取镜像，此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包，镜像中已经默认安装了昇腾算子库 CANN-8.0.0。
```bash
# 适用于 X86 架构
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-910b-base-x86_64-gcc84
# 适用于 Aarch64 架构
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-910b-base-aarch64-gcc84
```
* 参考如下命令启动容器，ASCEND_RT_VISIBLE_DEVICES 指定可见的 NPU 卡号
```bash
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-910b-base-$(uname -m)-gcc84 /bin/bash
```
## 2、安装paddle包
* 下载安装 wheel 安装包
```bash
# 注意需要先安装飞桨 cpu 版本
python -m pip install paddlepaddle==3.0.0.dev20250430 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu
python -m pip install paddle-custom-npu -i https://www.paddlepaddle.org.cn/packages/nightly/npu
```
* CANN-8.0.RC2 对 numpy 和 opencv 部分版本不支持，建议安装指定版本
```bash
python -m pip install numpy==1.26.4
python -m pip install opencv-python==3.4.18.65
```
* arm机器上需要设置环境变量（x86环境无需设置）
```bash
# 解决libgomp在arm机器上报错
# "libgomp cannot allocate memory in static TLS block"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD
```
* 验证安装包安装完成之后，运行如下命令
```bash
python -c "import paddle; paddle.utils.run_check()"
```
预期得到如下输出结果

```
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 npu.
PaddlePaddle works well on 8 npus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
