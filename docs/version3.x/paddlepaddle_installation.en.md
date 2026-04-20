---
comments: true
---

# PaddlePaddle Framework Installation

This document explains how to install PaddlePaddle. The following scenarios usually require the PaddlePaddle framework to be installed first:

- using the PaddlePaddle framework as the inference engine for pipeline/model inference;
- performing development tasks such as model training and export.

## 1. Install PaddlePaddle with Docker

<b>If you install via Docker</b>, use the following commands with the official PaddlePaddle Docker images to create a container named `paddleocr` and mount the current working directory to `/paddle` inside the container.

If your Docker version is >= 19.03, run:

```bash
# For CPU users:
docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0 /bin/bash

# For GPU users:
# GPU version, requires driver version >= 450.80.02 (Linux) or >= 452.39 (Windows)
docker run --gpus all --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash

# GPU version, requires driver version >= 550.54.14 (Linux) or >= 550.54.14 (Windows)
docker run --gpus all --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5 /bin/bash
```

If your Docker version is <= 19.03 but >= 17.06, run:

<details><summary>Click to expand</summary>

<pre><code class="language-bash"># For CPU users:
docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0 /bin/bash

# For GPU users:
# CUDA 11.8 users
nvidia-docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash

# CUDA 12.6 users
nvidia-docker run --name paddleocr -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5 /bin/bash
</code></pre>
</details>

If your Docker version is <= 17.06, please upgrade Docker first.

For more official PaddlePaddle Docker images, see the [PaddlePaddle website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/docker/linux-docker.html).

## 2. Install PaddlePaddle with pip

<b>If you install via pip</b>, use the following commands to install PaddlePaddle in the current environment:

```bash
# CPU version
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# GPU version, requires driver version >= 450.80.02 (Linux) or >= 452.39 (Windows)
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# GPU version, requires driver version >= 550.54.14 (Linux) or >= 550.54.14 (Windows)
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

> ❗ <b>Note</b>: You do not need to pay attention to the physical machine's CUDA version. You only need to care about the GPU driver version. For more PaddlePaddle wheel versions, see the [PaddlePaddle website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip.html).

After installation, use the following command to verify whether PaddlePaddle is installed successfully:

```bash
python -c "import paddle; print(paddle.__version__)"
```

If the installation succeeds, it will output a version number like:

```bash
3.2.0
```

## 3. Install PaddlePaddle wheel packages for NVIDIA 50-series GPUs on Windows

PaddlePaddle installed using the methods above does not properly support NVIDIA 50-series GPUs on Windows. Therefore, we provide specially adapted PaddlePaddle packages for this hardware environment. Please choose the corresponding wheel file according to your Python version.

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

The currently released PaddlePaddle wheel packages for Windows 50-series GPUs still have known issues in text-recognition model training, and related support is still being adapted and improved.
