---
comments: true
---

# 飞桨框架安装

本文档说明如何安装 PaddlePaddle。以下场景通常需要先安装飞桨框架：

- 在进行产线/模型推理时，使用飞桨框架作为推理引擎；
- 进行模型训练、导出等开发工作。

## 1. 基于 Docker 安装飞桨

<b>若您通过 Docker 安装</b>，请参考下述命令，使用飞桨框架官方 Docker 镜像，创建一个名为 `paddleocr` 的容器，并将当前工作目录映射到容器内的 `/paddle` 目录。

若您使用的 Docker 版本 >= 19.03，请执行：

```bash
# 对于 CPU 用户:
docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0 /bin/bash

# 对于 GPU 用户:
# GPU 版本，需显卡驱动程序版本 ≥450.80.02（Linux）或 ≥452.39（Windows）
docker run --gpus all --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash

# GPU 版本，需显卡驱动程序版本 ≥550.54.14（Linux）或 ≥550.54.14（Windows）
docker run --gpus all --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5 /bin/bash
```

若您使用的 Docker 版本 <= 19.03 但 >= 17.06，请执行：

<details><summary>点击展开</summary>

<pre><code class="language-bash"># 对于 CPU 用户:
docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0 /bin/bash

# 对于 GPU 用户:
# CUDA11.8 用户
nvidia-docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash

# CUDA12.6 用户
nvidia-docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5 /bin/bash
</code></pre>
</details>

若您使用的 Docker 版本 <= 17.06，请升级 Docker 版本。

更多飞桨官方 Docker 镜像请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html)。

## 2. 基于 pip 安装飞桨

<b>若您通过 pip 安装</b>，请参考下述命令，用 pip 在当前环境中安装 PaddlePaddle：

```bash
# CPU 版本
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# GPU 版本，需显卡驱动程序版本 ≥450.80.02（Linux）或 ≥452.39（Windows）
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# GPU 版本，需显卡驱动程序版本 ≥550.54.14（Linux）或 ≥550.54.14（Windows）
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

> ❗ <b>注</b>：无需关注物理机上的 CUDA 版本，只需关注显卡驱动程序版本。更多飞桨 Wheel 版本请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

安装完成后，使用以下命令验证 PaddlePaddle 是否安装成功：

```bash
python -c "import paddle; print(paddle.__version__)"
```

如果已安装成功，将输出如下版本号：

```bash
3.2.0
```

## 3. Windows 系统适配 NVIDIA 50 系显卡的 PaddlePaddle wheel 包安装

通过以上方式安装的 PaddlePaddle 在 Windows 操作系统下无法正常支持 NVIDIA 50 系显卡。因此，我们提供了专门适配该硬件环境的 PaddlePaddle 安装包。请根据您的 Python 版本选择对应的 wheel 文件进行安装。

```bash
# python 3.9
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp39-cp39-win_amd64.whl

# python 3.10
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp310-cp310-win_amd64.whl

# python 3.11
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp311-cp311-win_amd64.whl

# python 3.12
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp312-cp312-win_amd64.whl
```

当前发布的适用于 Windows 系统 50 系显卡的 PaddlePaddle wheel 包，其文本识别模型的训练存在已知问题，相关功能仍在持续适配和完善中。
