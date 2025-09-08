---
comments: true
---

# High-Performance Inference

In real-world production environments, many applications have stringent performance requirements for deployment strategies, particularly regarding response speed, to ensure efficient system operation and a smooth user experience. PaddleOCR provides high-performance inference capabilities, allowing users to enhance model inference speed with a single click without worrying about complex configurations or underlying details. Specifically, PaddleOCR's high-performance inference functionality can:

- Automatically select an appropriate inference backend (e.g., Paddle Inference, OpenVINO, ONNX Runtime, TensorRT) based on prior knowledge and configure acceleration strategies (e.g., increasing the number of inference threads, setting FP16 precision inference).
- Automatically convert PaddlePaddle static graph models to ONNX format as needed to leverage better inference backends for acceleration.
- Perform inference using ONNX models.

This document primarily introduces the installation and usage methods for high-performance inference.

## 1. Prerequisites

### 1.1 Install High-Performance Inference Dependencies

Install the dependencies required for high-performance inference using the PaddleOCR CLI:

```bash
paddleocr install_hpi_deps {device_type}
```

The supported device types are:

- `cpu`: For CPU-only inference. Currently supports Linux systems, x86-64 architecture processors, and Python 3.8-3.12.
- `gpu`: For inference using either CPU or NVIDIA GPU. Currently supports Linux systems, x86-64 architecture processors, and Python 3.8-3.12. If you want to use the full high-performance inference capabilities, you also need to ensure that a compatible version of TensorRT is installed in your environment. Refer to the next subsection for detailed instructions.

Only one type of device dependency should exist in the same environment. For Windows systems, it is currently recommended to install within a Docker container or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) environment.

**It is recommended to use the official PaddlePaddle Docker image to install high-performance inference dependencies.** The corresponding images for each device type are as follows:

- `cpu`：`ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.1-paddlepaddle3.0.0-cpu`
- `gpu`：
    - CUDA 11.8：`ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.1-paddlepaddle3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6`
- `gpu`：
    - CUDA 12.6：`ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.1-paddlepaddle3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5`

**Notice:**

- **Currently, high-performance inference with CUDA 12.6 and cuDNN 9.5 only supports OpenVINO and ONNX Runtime backends, and does not yet support the TensorRT backend.**

### 1.2 Detailed GPU Environment Instructions

First, ensure that the environment has the required CUDA and cuDNN versions installed. Currently, PaddleOCR supports CUDA and cuDNN versions compatible with CUDA 11.8 + cuDNN 8.9 or CUDA 12.6 + cuDNN 9.5. Below are the installation instructions for CUDA and cuDNN:

- [Install CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [Install cuDNN 8.9](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-890/install-guide/index.html)
- [Install CUDA 12.6](https://developer.nvidia.com/cuda-12-6-0-download-archive)
- [Install cuDNN 9.5](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.5.0/installation/linux.html)

If using the official PaddlePaddle image, the CUDA and cuDNN versions in the image already meet the requirements, and no additional installation is needed.

If installing PaddlePaddle via pip, the relevant Python packages for CUDA and cuDNN will typically be installed automatically. In this case, **you still need to install the non-Python-specific CUDA and cuDNN versions.** It is also recommended to install CUDA and cuDNN versions that match the Python package versions in your environment to avoid potential issues caused by coexisting library versions. You can check the versions of the CUDA and cuDNN-related Python packages with the following commands:

```bash
# CUDA-related Python package versions
pip list | grep nvidia-cuda
# cuDNN-related Python package versions
pip list | grep nvidia-cudnn
```

Secondly, it is recommended to ensure that a compatible version of TensorRT is installed in the environment; otherwise, the Paddle Inference TensorRT subgraph engine will be unavailable, and the program may not achieve optimal inference performance. Currently, PaddleOCR only supports TensorRT 8.6.1.6 in the CUDA 11.8 environment. If using the official PaddlePaddle 3.0 Docker image, you can install the TensorRT wheel package with the following command:

```bash
python -m pip install /usr/local/TensorRT-*/python/tensorrt-*-cp310-none-linux_x86_64.whl
```

For other environments, refer to the [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html) to install TensorRT. Here is an example:

```bash
# Download the TensorRT tar file
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
# Extract the TensorRT tar file
tar xvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
# Install the TensorRT wheel package
python -m pip install TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl
# Add the absolute path of the TensorRT `lib` directory to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:TensorRT-8.6.1.6/lib"
```

## 2. Executing High-Performance Inference

For the PaddleOCR CLI, specify `--enable_hpi` as `True` to execute high-performance inference. For example:

```bash
paddleocr ocr --enable_hpi True ...
```

For the PaddleOCR Python API, set `enable_hpi` to `True` when initializing the pipeline or module object to enable high-performance inference when calling the inference method. For example:

```python
from paddleocr import PaddleOCR
pipeline = PaddleOCR(enable_hpi=True)
result = pipeline.predict(...)
```

## 3. Notes

1. For some models, the first execution of high-performance inference may take longer to complete the construction of the inference engine. Relevant information about the inference engine will be cached in the model directory after the first construction, and subsequent initializations can reuse the cached content to improve speed.

2. Currently, due to reasons such as not using static graph format models or the presence of unsupported operators, some models may not achieve inference acceleration.

3. During high-performance inference, PaddleOCR automatically handles the conversion of model formats and selects the optimal inference backend whenever possible. Additionally, PaddleOCR supports users specifying ONNX models. For information on converting PaddlePaddle static graph models to ONNX format, refer to [Obtaining ONNX Models](./obtaining_onnx_models.en.md).

4. The high-performance inference capabilities of PaddleOCR rely on PaddleX and its high-performance inference plugins. By passing in a custom PaddleX production line configuration file, you can configure the inference backend and other related settings. Please refer to [Using PaddleX Production Line Configuration Files](../paddleocr_and_paddlex.en.md#3-Using-PaddleX-Pipeline-Configuration-Files) and the [PaddleX High-Performance Inference Guide](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/high_performance_inference.html#22) to learn how to adjust the high-performance inference configurations.
