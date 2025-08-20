---
comments: true
---

# 安装

# 1. 安装飞桨框架

安装飞桨 PaddlePaddle 时，支持通过 Docker 安装和通过 pip 安装。

## 1.1 基于 Docker 安装飞桨
<b>若您通过 Docker 安装</b>，请参考下述命令，使用飞桨框架官方 Docker 镜像，创建一个名为 `paddleocr` 的容器，并将当前工作目录映射到容器内的 `/paddle` 目录：

若您使用的 Docker 版本 >= 19.03，请执行：

```bash
# 对于 cpu 用户:
docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0 /bin/bash

# 对于 gpu 用户:
# GPU 版本，需显卡驱动程序版本 ≥450.80.02（Linux）或 ≥452.39（Windows）
docker run --gpus all --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash

# GPU 版本，需显卡驱动程序版本 ≥545.23.06（Linux）或 ≥545.84（Windows）
docker run --gpus all --name paddleocr -v $PWD:/paddle  --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5 /bin/bash
```

* 若您使用的 Docker 版本 <= 19.03 但 >= 17.06，请执行：

<details><summary> 点击展开</summary>

<pre><code class="language-bash"># 对于 cpu 用户:
docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0 /bin/bash

# 对于 gpu 用户:
# CUDA11.8 用户
nvidia-docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash

# CUDA12.3 用户
nvidia-docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5 /bin/bash
</code></pre></details>

* 若您使用的 Docker 版本 <= 17.06，请升级 Docker 版本。

* 注：更多飞桨官方 docker 镜像请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html)。

## 1.2 基于 pip 安装飞桨
<b>若您通过 pip 安装</b>，请参考下述命令，用 pip 在当前环境中安装飞桨 PaddlePaddle：

```bash
# CPU 版本
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# GPU 版本，需显卡驱动程序版本 ≥450.80.02（Linux）或 ≥452.39（Windows）
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# GPU 版本，需显卡驱动程序版本 ≥550.54.14（Linux）或 ≥550.54.14（Windows）
 python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

> ❗ <b>注</b>：无需关注物理机上的 CUDA 版本，只需关注显卡驱动程序版本。更多飞桨 Wheel 版本请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

安装完成后，使用以下命令可以验证 PaddlePaddle 是否安装成功：

```bash
python -c "import paddle; print(paddle.__version__)"
```
如果已安装成功，将输出以下内容：

```bash
3.0.0
```

## 1.3 Windows 系统适配 NVIDIA 50 系显卡的 PaddlePaddle wheel 包安装

通过以上方式安装的 PaddlePaddle 在 Windows 操作系统下无法正常支持 NVIDIA 50 系显卡。因此，我们提供了专门适配该硬件环境的 PaddlePaddle 安装包。请根据您的 Python 版本选择对应的 wheel 文件进行安装。

```bash
# python 3.9
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp39-cp39-win_amd64.whl

# python 3.10
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp310-cp310-win_amd64.whl

# python 3.11
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp311-cp311-win_amd64.whl

# python 3.12
python -m https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp312-cp312-win_amd64.whl
```

**注：** 当前发布的适用于 Windows 系统 50 系显卡的 PaddlePaddle wheel 包，其文本识别模型的训练存在已知问题，相关功能仍在持续适配和完善中。

# 2. 安装 PaddleOCR

如果只希望使用 PaddleOCR 的推理功能，请参考 [安装推理包](#21-安装推理包)；如果希望进行模型训练、导出等，请参考 [安装训练依赖](#22-安装训练依赖)。在同一环境中安装推理包和训练依赖是允许的，无需进行环境隔离。

## 2.1 安装推理包

从 PyPI 安装最新版本 PaddleOCR 推理包：

```bash
python -m pip install paddleocr
```

或者从源码安装（默认为开发分支）：

```bash
python -m pip install "git+https://github.com/PaddlePaddle/PaddleOCR.git"
```

## 2.2 安装训练依赖

要进行模型训练、导出等，需要首先将仓库克隆到本地：

```bash
# 推荐方式
git clone https://github.com/PaddlePaddle/PaddleOCR

# （可选）切换到指定分支
git checkout release/3.0

# 如果因为网络问题无法克隆成功，也可选择使用码云上的仓库：
git clone https://gitee.com/paddlepaddle/PaddleOCR

# 注：码云托管代码可能无法实时同步本 GitHub 项目更新，存在3~5天延时，请优先使用推荐方式。
```

执行如下命令安装依赖：

```bash
python -m pip install -r requirements.txt
```
