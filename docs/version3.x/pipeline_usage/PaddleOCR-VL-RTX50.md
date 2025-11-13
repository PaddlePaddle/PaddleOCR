---
comments: true
---

# PaddleOCR-VL-RTX50 环境配置教程

本教程是 NVIDIA RTX 50 系 GPU 的环境配置教程，目的是完成相关的环境配置，环境配置完毕后请参考 [PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 使用 PaddleOCR-VL。

教程开始前，**请确认您的 NVIDIA 驱动支持 CUDA 12.9 或以上版本**。

## 1. 环境准备

此步骤主要介绍如何搭建 PaddleOCR-VL 的运行环境，有以下两种方式，任选一种即可：

- 方法一：使用官方 Docker 镜像（暂不支持，正在适配中）。

- 方法二：手动安装 PaddlePaddle 和 PaddleOCR。

### 1.1 方法一：使用 Docker 镜像

暂不支持，正在适配中。

### 1.2 方法二：手动安装 PaddlePaddle 和 PaddleOCR

如果您无法使用 Docker，也可以手动安装 PaddlePaddle 和 PaddleOCR。要求 Python 版本为 3.8–3.12。

**我们强烈推荐您在虚拟环境中安装 PaddleOCR-VL，以避免发生依赖冲突。** 例如，使用 Python venv 标准库创建虚拟环境：

```shell
# 创建虚拟环境
python -m venv .venv_paddleocr
# 激活环境
source .venv_paddleocr/bin/activate
```

执行如下命令完成安装：

```shell
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
python -m pip install -U "paddleocr[doc-parser]"
# 对于 Linux 系统，执行：
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
# 对于Windows 系统，执行：
python -m pip install https://xly-devops.cdn.bcebos.com/safetensors-nightly/safetensors-0.6.2.dev0-cp38-abi3-win_amd64.whl
```

> **请注意安装 3.2.1 及以上版本的飞桨框架，同时安装特殊版本的 safetensors。**

## 2. 快速开始

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md)相同章节。

## 3. 使用推理加速框架提升 VLM 推理性能

默认配置下的推理性能未经过充分优化，可能无法满足实际生产需求。此步骤主要介绍如何使用 vLLM 和 SGLang 推理加速框架来提升 PaddleOCR-VL 的推理性能。

### 3.1 启动 VLM 推理服务

启动 VLM 推理服务有以下两种方式，任选一种即可：

- 方法一：使用官方 Docker 镜像启动服务（暂不支持，正在适配中）。

- 方法二：通过 PaddleOCR CLI 手动安装依赖后启动服务。

#### 3.1.1 方法一：使用 Docker 镜像

暂不支持，正在适配中。

#### 3.1.2 方法二：通过 PaddleOCR CLI 安装和使用

由于推理加速框架可能与飞桨框架存在依赖冲突，建议在虚拟环境中安装。以 vLLM 为例：

```shell
# 如果当前存在已激活的虚拟环境，先通过 `deactivate` 取消激活
# 创建虚拟环境
python -m venv .venv_vlm
# 激活环境
source .venv_vlm/bin/activate
# 安装 PaddleOCR
python -m pip install "paddleocr[doc-parser]"
# 安装推理加速服务依赖
paddleocr install_genai_server_deps vllm
python -m pip install flash-attn==2.8.3
```

> `paddleocr install_genai_server_deps` 命令在执行过程中可能需要使用 nvcc 等 CUDA 编译工具。如果您的环境中没有这些工具或者安装时间过长，可以从 [此仓库](https://github.com/mjun0812/flash-attention-prebuild-wheels) 获取 FlashAttention 的预编译版本，例如执行 `python -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl`。

`paddleocr install_genai_server_deps` 命令用法：

```shell
paddleocr install_genai_server_deps <推理加速框架名称>
```

当前支持的框架名称为 `vllm` 和 `sglang`，分别对应 vLLM 和 SGLang。

安装完成后，可通过 `paddleocr genai_server` 命令启动服务：

```shell
paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8118
```

该命令支持的参数如下：

| 参数                 | 说明                        |
| ------------------ | ------------------------- |
| `--model_name`     | 模型名称                      |
| `--model_dir`      | 模型目录                      |
| `--host`           | 服务器主机名                    |
| `--port`           | 服务器端口号                    |
| `--backend`        | 后端名称，即使用的推理加速框架名称，可选 `vllm` 或 `sglang` |
| `--backend_config` | 可指定 YAML 文件，包含后端配置        |

### 3.2 客户端使用方法

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

## 4. 服务化部署

此步骤主要介绍如何将 PaddleOCR-VL 部署为服务并调用，有以下两种方式，任选一种即可：

- 方法一：使用 Docker Compose 部署（暂不支持，正在适配中）。

- 方法二：手动安装依赖部署。

请注意，本节所介绍 PaddleOCR-VL 服务与上一节中的 VLM 推理服务有所区别：后者仅负责完整流程中的一个环节（即 VLM 推理），并作为前者的底层服务被调用。

### 4.1 方法一：使用 Docker Compose 部署

暂不支持，正在适配中。

### 4.2 方法二：手动安装依赖部署

执行以下命令，通过 PaddleX CLI 安装服务化部署插件：

```shell
paddlex --install serving
```

然后，使用 PaddleX CLI 启动服务器：

```shell
paddlex --serve --pipeline PaddleOCR-VL
```

启动后将看到类似如下输出，服务器默认监听 **8080** 端口：

```text
INFO:     Started server process [63108]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

与服务化部署相关的命令行参数如下：

<table>
<thead>
<tr>
<th>名称</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>--pipeline</code></td>
<td>PaddleX 产线注册名或产线配置文件路径。</td>
</tr>
<tr>
<td><code>--device</code></td>
<td>产线部署设备。默认情况下，若 GPU 可用则使用 GPU，否则使用 CPU。</td>
</tr>
<tr>
<td><code>--host</code></td>
<td>服务器绑定的主机名或 IP 地址，默认为 <code>0.0.0.0</code>。</td>
</tr>
<tr>
<td><code>--port</code></td>
<td>服务器监听的端口号，默认为 <code>8080</code>。</td>
</tr>
<tr>
<td><code>--use_hpip</code></td>
<td>启用高性能推理模式。请参考高性能推理文档了解更多信息。</td>
</tr>
<tr>
<td><code>--hpi_config</code></td>
<td>高性能推理配置。请参考高性能推理文档了解更多信息。</td>
</tr>
</tbody>
</table>

如需调整产线相关配置（如模型路径、批处理大小、部署设备等），可参考 4.4 小节。

### 4.3 客户端调用方式

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

### 4.4 产线配置调整说明

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

## 5. 模型微调

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。
