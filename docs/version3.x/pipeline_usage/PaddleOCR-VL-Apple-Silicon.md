---
comments: true
---

# PaddleOCR-VL Apple Silicon 环境配置教程

本教程是 PaddleOCR-VL Apple Silicon 的环境配置教程，目的是完成相关的环境配置，环境配置完毕后请参考 [PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 使用 PaddleOCR-VL。

Apple Silicon 包括但不限于以下几种：

- Apple M1
- Apple M2
- Apple M3
- Apple M4

目前 PaddleOCR-VL 已在 Apple M4 上完成精度验证；鉴于硬件环境的多样性，其他 Apple Silicon 的兼容性尚未验证。我们诚挚欢迎社区用户在不同硬件上进行测试并反馈您的运行结果。

## 1. 环境准备

**我们强烈推荐您在虚拟环境中安装 PaddleOCR-VL，以避免发生依赖冲突。** 例如，使用 Python venv 标准库创建虚拟环境：

```shell
# 创建虚拟环境
python -m venv .venv_paddleocr
# 激活环境
source .venv_paddleocr/bin/activate
```

执行如下命令完成安装：

```shell
python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -U "paddleocr[doc-parser]"
```

> **请注意安装 3.2.1 及以上版本的飞桨框架。**

## 2. 快速开始

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md)相同章节。

## 3. 使用推理加速框架提升 VLM 推理性能

默认配置下的推理性能未经过充分优化，可能无法满足实际生产需求。此步骤主要介绍如何使用 MLX-VLM 推理加速框架来提升 PaddleOCR-VL 的推理性能。

### 3.1 启动 VLM 推理服务

安装 MLX-VLM 推理框架：

```shell
python -m pip install -U mlx-vlm
```

启动 MLX-VLM 推理服务：

```shell
mlx_vlm.server --port 8111
```

### 3.2 客户端使用方法

#### 3.2.1 CLI 调用

可通过 `--vl_rec_backend` 指定后端类型（`mlx-vlm-server`），通过 `--vl_rec_server_url` 指定服务地址，通过 `--vl_rec_api_model_name` 指定 huggingface repo id 或服务端模型权重路径，例如：

```shell
paddleocr doc_parser \
  --input paddleocr_vl_demo.png \
  --vl_rec_backend mlx-vlm-server \
  --vl_rec_server_url http://localhost:8111/ \
  --vl_rec_api_model_name PaddlePaddle/PaddleOCR-VL-1.5
```

#### 3.2.2 Python API 调用

创建 `PaddleOCRVL` 对象时传入 `vl_rec_backend` 指定后端类型， `vl_rec_server_url` 参数指定服务地址，`vl_rec_api_model_name` 指定 huggingface repo id 或服务端模型权重路径，例如：

```python
pipeline = PaddleOCRVL(
    vl_rec_backend="mlx-vlm-server", 
    vl_rec_server_url="http://localhost:8111/",
    vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
)
```

### 3.3 性能调优

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

## 4. 服务化部署

目前仅支持**手动部署**方式，请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) **4.2 方法二：手动部署**。

### 4.3 客户端调用方式

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

### 4.4 产线配置调整说明

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

## 5. 模型微调

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。
