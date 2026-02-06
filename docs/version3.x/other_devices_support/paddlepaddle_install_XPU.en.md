---
comments: true
---

# Kunlun XPU PaddlePaddle Installation Tutorial

Currently, PaddleOCR supports Kunlun R200/R300 and other chips. Considering environmental differences, we recommend using the <b>Kunlun XPU development image officially released by PaddlePaddle</b>, which is pre-installed with the Kunlun basic runtime environment library (XRE).

## 1. Docker Environment Preparation
Pull the image. This image is only for the development environment and does not include a pre-compiled PaddlePaddle installation package.

```bash
docker pull registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310 # For X86 architecture
docker pull registry.baidubce.com/device/paddle-xpu:kylinv10-aarch64-gcc82-py310 # For ARM architecture
```
Refer to the following command to start the container:

```bash
docker run -it --name=xxx -m 81920M --memory-swap=81920M \
    --shm-size=128G --privileged --net=host \
    -v $(pwd):/workspace -w /workspace \
    registry.baidubce.com/device/paddle-xpu:$(uname -m)-py310 bash
```

## 2. Install Paddle Package
Currently, Python3.10 wheel installation packages are provided. If you have a need for other Python versions, you can refer to the [PaddlePaddle official documentation](https://www.paddlepaddle.org.cn/en/install/quick) to compile and install them yourself.

Install the Python3.10 wheel installation package:

```bash
pip install https://paddle-whl.bj.bcebos.com/paddlex/xpu/paddlepaddle_xpu-2.6.1-cp310-cp310-linux_x86_64.whl # For X86 architecture
pip install https://paddle-whl.bj.bcebos.com/paddlex/xpu/paddlepaddle_xpu-2.6.1-cp310-cp310-linux_aarch64.whl # For ARM architecture
```

Verify the installation package. After installation, run the following command:

```bash
python -c "import paddle; paddle.utils.run_check()"
```

The expected output is:

```
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
