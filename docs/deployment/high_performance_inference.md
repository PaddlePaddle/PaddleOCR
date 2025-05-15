# 高性能推理

在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。PaddleOCR 提供高性能推理能力，让用户无需关注复杂的配置和底层细节，一键提升模型的推理速度。具体而言，PaddleOCR 的高性能推理功能能够：

- 结合先验知识自动选择合适的推理后端（Paddle Inference、OpenVINO、ONNX Runtime、TensorRT等），并配置加速策略（如增大推理线程数、设置 FP16 精度推理）；
- 根据需要自动将飞桨静态图模型转换为 ONNX 格式，以使用更优的推理后端实现加速。

本文档主要介绍高性能推理功能的安装与使用方法。

## 1. 前置条件

## 1.1 安装高性能推理依赖

通过 PaddleOCR CLI 安装高性能推理所需依赖：

```bash
paddleocr install_hpi_deps {设备类型}
```

支持的设备类型包括：

- `cpu`：仅使用 CPU 推理。目前支持 Linux 系统、x86-64 架构处理器、Python 3.8-3.12。
- `gpu`：使用 CPU 或 NVIDIA GPU 推理。目前支持 Linux 系统、x86-64 架构处理器、Python 3.8-3.12。请查看下一小节的详细说明。

同一环境中只应该存在一种设备类型的依赖。对于 Windows 系统，目前建议在 Docker 容器或者 [WSL](https://learn.microsoft.com/zh-cn/windows/wsl/install) 环境中安装。

**推荐使用飞桨官方 Docker 镜像安装高性能推理依赖。** 各设备类型对应的镜像如下：

- `cpu`：`ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0`
- `gpu`：
    - CUDA 11.8：`ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6`

## 1.2 GPU 环境详细说明

首先，需要确保环境中安装有符合要求的 CUDA 与 cuDNN。目前 PaddleOCR 仅支持与 CUDA 11.8 + cuDNN 8.9 兼容的 CUDA 和 cuDNN版本。以下分别是 CUDA 11.8 和 cuDNN 8.9 的安装说明文档：

- [安装 CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [安装 cuDNN 8.9](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-890/install-guide/index.html)

如果使用飞桨官方镜像，则镜像中的 CUDA 和 cuDNN 版本已经是满足要求的，无需额外安装。

如果通过 pip 安装飞桨，通常 CUDA、cuDNN 的相关 Python 包将被自动安装。在这种情况下，**仍需要通过安装非 Python 专用的 CUDA 与 cuDNN**。同时，建议安装的 CUDA 和 cuDNN 版本与环境中存在的 Python 包版本保持一致，以避免不同版本的库共存导致的潜在问题。可以通过如下方式可以查看 CUDA 和 cuDNN 相关 Python 包的版本：

```bash
# CUDA 相关 Python 包版本
pip list | grep nvidia-cuda
# cuDNN 相关 Python 包版本
pip list | grep nvidia-cudnn
```

其次，需确保环境中安装有符合要求的 TensorRT。目前 PaddleOCR 仅支持 TensorRT 8.6.1.6。如果使用飞桨官方镜像，可执行如下命令安装 TensorRT wheel 包：

```bash
python -m pip install /usr/local/TensorRT-*/python/tensorrt-*-cp310-none-linux_x86_64.whl
```

对于其他环境，请参考 [TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html) 安装 TensorRT。示例如下：

```bash
# 下载 TensorRT tar 文件
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
# 解压 TensorRT tar 文件
tar xvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
# 安装 TensorRT wheel 包
python -m pip install TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl
# 添加 TensorRT 的 `lib` 目录的绝对路径到 LD_LIBRARY_PATH 中
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:TensorRT-8.6.1.6/lib"
```

## 2. 执行高性能推理

对于 PaddleOCR CLI，指定 `--enable_hpi` 为 `True` 即可执行高性能推理。例如：

```bash
paddleocr ocr --enable_hpi True ...
```

对于 PaddleOCR Python API，在初始化产线对象或者模块对象时，设置 `enable_hpi` 为 `True` 即可在调用推理方法时执行高性能推理。例如：

```python
from paddleocr import PaddleOCR
pipeline = PaddleOCR(enable_hpi=True)
result = pipeline.predict(...)
```

## 3. 说明

1. 对于部分模型，在首次执行高性能推理时，可能需要花费较长时间完成推理引擎的构建。推理引擎相关信息将在第一次构建完成后被缓存在模型目录，后续可复用缓存中的内容以提升初始化速度。
2. 目前，由于使用的不是静态图格式模型、存在不支持算子等原因，部分模型可能无法获得推理加速。
3. 在进行高性能推理时，PaddleOCR 会自动处理模型格式的转换，并尽可能选择最优的推理后端。同时，PaddleOCR 也支持用户指定 ONNX 模型。有关如何飞桨静态图模型转换为 ONNX 格式，可参考 [获取 ONNX 模型](./obtaining_onnx_models.md)。
4. PaddleOCR 的高性能推理能力依托于 PaddleX 及其高性能推理插件。通过传入自定义 PaddleX 产线配置文件，可以对推理后端等进行配置。请参考 [使用 PaddleX 产线配置文件](../paddleocr_and_paddlex.md#3-使用-paddlex-产线配置文件) 和 [PaddleX 高性能推理指南](https://paddlepaddle.github.io/PaddleX/3.0/pipeline_deploy/high_performance_inference.html#22) 了解如何调整高性能推理配置。
