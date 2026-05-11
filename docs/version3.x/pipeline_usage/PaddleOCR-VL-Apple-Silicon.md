---
comments: true
---

# PaddleOCR-VL Apple Silicon 使用教程

> INFO:
> 除非另有说明，本教程中提到的 “PaddleOCR-VL” 均指 PaddleOCR-VL 系列模型（如 PaddleOCR-VL-1.5 等）；若特指 PaddleOCR-VL v1 版本，将另行明确标注。

本教程是 PaddleOCR-VL 在 Apple Silicon 上的使用指南，涵盖了从环境准备到服务化部署的完整流程。

Apple Silicon 包括但不限于以下几种：

- Apple M1
- Apple M2
- Apple M3
- Apple M4

目前 PaddleOCR-VL 已在 Apple M4 上完成精度验证；鉴于硬件环境的多样性，其他 Apple Silicon 的兼容性尚未验证。我们诚挚欢迎社区用户在不同硬件上进行测试并反馈您的运行结果。

## 本硬件支持的使用目标

请在本硬件教程中按下表继续阅读。

| 目标 | 本硬件上的支持情况 | 从哪里开始阅读 |
| --- | --- | --- |
| 本地直接推理 | 支持 | 阅读第 1 节“环境准备”和第 2 节“快速开始”。 |
| 客户端 + VLM 推理服务 | 支持 | 先完成本地直接推理，再阅读第 3 节“使用 VLM 推理服务提升推理性能”。 |
| 完整 API 服务 | 仅支持手动部署 | 先完成第 1 节“环境准备”，再阅读第 4.1 节“手动部署”；然后继续阅读第 4.2 节客户端调用部分和第 4.3 节产线配置调整部分。 |
| 模型微调 | 支持 | 阅读第 5 节“模型微调”。 |

如果你只是想先确认本硬件支持哪些推理方式，请参考主教程中的 [PaddleOCR-VL 推理方式与硬件支持矩阵](./PaddleOCR-VL.md#paddleocr-vl-对推理设备的支持情况)。

## 1. 环境准备

**当前硬件支持的环境准备方式**

| 环境准备方式 | 状态 | 说明 |
| --- | --- | --- |
| 官方 Docker 镜像 | 当前不支持 | 当前硬件不支持该路径。 |
| 手动安装 PaddlePaddle 和 PaddleOCR | 支持并提供步骤 | 请继续阅读本节。 |

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

请参考[PaddleOCR-VL 使用教程 - 2. 快速开始](./PaddleOCR-VL.md#2)。

## 3. 使用 VLM 推理服务提升推理性能

默认配置下的推理性能未经过充分优化，可能无法满足实际生产需求。此步骤主要介绍如何通过 VLM 推理服务提升 PaddleOCR-VL 的推理性能。在当前硬件文档中，示例使用 MLX-VLM 作为 VLM 推理服务后端。

### 3.1 启动 VLM 推理服务

> IMPORTANT:
> 按照本节说明启动的服务仅负责 PaddleOCR-VL 流程中的 VLM 推理环节，不提供完整的端到端文档解析 API。强烈不建议直接通过 HTTP 请求或使用 OpenAI 客户端调用该服务处理文档图像。若您需要部署具备 PaddleOCR-VL 完整能力的服务，请参考后文的服务化部署部分。

**当前硬件支持的启动方式**

| 启动方式 | 状态 | 说明 |
| --- | --- | --- |
| 官方 Docker 镜像 | 当前不支持 | 当前硬件不支持该路径。 |
| 通过 PaddleOCR CLI 安装依赖后启动 | 当前不支持 | 当前硬件不支持该路径。 |
| 直接使用推理加速框架启动 | 支持并提供步骤 | 本节提供 MLX-VLM 的启动步骤。 |

安装 MLX-VLM 推理框架（v0.3.11以上版本）：

```shell
python -m pip install "mlx-vlm>=0.3.11"
```

启动 MLX-VLM 推理服务：

```shell
mlx_vlm.server --port 8111
```

### 3.2 客户端使用方法

以下调用方式适用于已启动的 MLX-VLM 推理服务。

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

请参考[PaddleOCR-VL 使用教程 - 3.3 性能调优](./PaddleOCR-VL.md#33)。

## 4. 服务化部署

**当前硬件支持的部署方式**

| 部署方式 | 状态 | 说明 |
| --- | --- | --- |
| Docker Compose 部署 | 当前不支持 | 当前硬件仅支持手动部署路径。 |
| 手动部署 | 支持（步骤见主教程） | 请先完成第 1 节“环境准备”，再继续阅读第 4.1 节，然后参考主教程中的共享手动部署说明。 |

### 4.1 手动部署

请先完成第 1 节“环境准备”，再参考 [PaddleOCR-VL 使用教程 - 4.2 方法二：手动部署](./PaddleOCR-VL.md#42)。

### 4.2 客户端调用方式

请参考[PaddleOCR-VL 使用教程 - 4.3 客户端调用方式](./PaddleOCR-VL.md#43)。

### 4.3 产线配置调整说明

请参考[PaddleOCR-VL 使用教程 - 4.4 产线配置调整说明](./PaddleOCR-VL.md#44)。

## 5. 模型微调

请参考[PaddleOCR-VL 使用教程 - 5. 模型微调](./PaddleOCR-VL.md#5)。
