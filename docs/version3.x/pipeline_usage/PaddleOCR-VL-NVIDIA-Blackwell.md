---
comments: true
---

# PaddleOCR-VL NVIDIA Blackwell 架构 GPU 环境配置教程

本教程是 NVIDIA Blackwell 架构 GPU 的环境配置教程，目的是完成相关的环境配置，环境配置完毕后请参考 [PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 使用 PaddleOCR-VL。

NVIDIA Blackwell 架构 GPU 包括但不限于以下几种：

- RTX 5090
- RTX 5080
- RTX 5070、RTX 5070 Ti
- RTX 5060、RTX 5060 Ti
- RTX 5050

教程开始前，**请确认您的 NVIDIA 驱动支持 CUDA 12.9 或以上版本**。

## 1. 环境准备

此步骤主要介绍如何搭建 PaddleOCR-VL 的运行环境，有以下两种方式，任选一种即可：

- 方法一：使用官方 Docker 镜像。

- 方法二：手动安装 PaddlePaddle 和 PaddleOCR。

### 1.1 方法一：使用 Docker 镜像

我们推荐使用官方 Docker 镜像（要求 Docker 版本 >= 19.03，机器装配有 GPU 且 NVIDIA 驱动支持 CUDA 12.9 或以上版本）：

```shell
docker run \
    -it \
    --gpus all \
    --network host \
    --user root \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-gpu-sm120 \
    /bin/bash
# 在容器中调用 PaddleOCR CLI 或 Python API
```

如果您希望在无法连接互联网的环境中使用 PaddleOCR-VL，请将上述命令中的 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-gpu-sm120`（镜像大小约为 10 GB）更换为离线版本镜像 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-gpu-sm120-offline`（镜像大小约为 12 GB）。

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
# 注意这里安装的是 cu129 的 PaddlePaddle
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
python -m pip install -U "paddleocr[doc-parser]"
# 对于 Linux 系统，执行：
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
# 对于 Windows 系统，执行：
python -m pip install https://xly-devops.cdn.bcebos.com/safetensors-nightly/safetensors-0.6.2.dev0-cp38-abi3-win_amd64.whl
```

> **请注意安装 3.2.1 及以上版本的飞桨框架，同时安装特殊版本的 safetensors。**

## 2. 快速开始

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md)相同章节。

## 3. 使用推理加速框架提升 VLM 推理性能

默认配置下的推理性能未经过充分优化，可能无法满足实际生产需求。此步骤主要介绍如何使用 vLLM 和 SGLang 推理加速框架来提升 PaddleOCR-VL 的推理性能。

### 3.1 启动 VLM 推理服务

启动 VLM 推理服务有以下两种方式，任选一种即可：

- 方法一：使用官方 Docker 镜像启动服务。

- 方法二：通过 PaddleOCR CLI 手动安装依赖后启动服务。

#### 3.1.1 方法一：使用 Docker 镜像

PaddleOCR 提供了 Docker 镜像，用于快速启动 vLLM 推理服务。可使用以下命令启动服务（要求 Docker 版本 >= 19.03，机器装配有 GPU 且 NVIDIA 驱动支持 CUDA 12.9 或以上版本）：

```shell
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-gpu-sm120 \
    paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend vllm
```

如果您希望在无法连接互联网的环境中启动服务，请将上述命令中的 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-gpu-sm120`（镜像大小约为 12 GB）更换为离线版本镜像 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-gpu-sm120-offline`（镜像大小约为 14 GB）。

启动 vLLM 推理服务时，我们提供了一套默认参数设置。如果您有调整显存占用等更多参数的需求，可以自行配置更多参数。请参考 [3.3.1 服务端参数调整](#331-服务端参数调整) 创建配置文件，然后将该文件挂载到容器中，并在启动服务的命令中使用 `backend_config` 指定配置文件，例如：

```shell
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    -v vllm_config.yml:/tmp/vllm_config.yml \  
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-gpu-sm120 \
    paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend vllm --backend_config /tmp/vllm_config.yml
```

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

- 方法一：使用 Docker Compose 部署。

- 方法二：手动安装依赖部署。

请注意，本节所介绍 PaddleOCR-VL 服务与上一节中的 VLM 推理服务有所区别：后者仅负责完整流程中的一个环节（即 VLM 推理），并作为前者的底层服务被调用。

### 4.1 方法一：使用 Docker Compose 部署

1. 从 [此处](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/compose.yaml) 复制内容保存为 `compose.yaml` 文件。

2. 复制以下内容并保存为 `.env` 文件：

    ```
    API_IMAGE_TAG_SUFFIX=latest-gpu-sm120-offline
    VLM_BACKEND=vllm
    VLM_IMAGE_TAG_SUFFIX=latest-gpu-sm120-offline
    ```

3. 在 `compose.yaml` 和 `.env` 文件所在目录下执行以下命令启动服务器，默认监听 **8080** 端口：

    ```shell
    # 必须在 compose.yaml 和 .env 文件所在的目录中执行
    docker compose up
    ```

    启动后将看到类似如下输出：

    ```text
    paddleocr-vl-api             | INFO:     Started server process [1]
    paddleocr-vl-api             | INFO:     Waiting for application startup.
    paddleocr-vl-api             | INFO:     Application startup complete.
    paddleocr-vl-api             | INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
    ```

此方式基于 vLLM 框架对 VLM 推理进行加速，更适合生产环境部署。

此外，使用此方式启动服务器后，除拉取镜像外，无需连接互联网。如需在离线环境中部署，可先在联网机器上拉取 Compose 文件中涉及的镜像，导出并传输至离线机器中导入，即可在离线环境下启动服务。

Docker Compose 通过读取 `.env` 和 `compose.yaml` 文件中配置，先后启动 2 个容器，分别运行底层 VLM 推理服务，以及 PaddleOCR-VL 服务（产线服务）。

`.env` 文件中包含的各环境变量含义如下：

- `API_IMAGE_TAG_SUFFIX`：启动产线服务使用的镜像的标签后缀。
- `VLM_BACKEND`：VLM 推理后端。
- `VLM_IMAGE_TAG_SUFFIX`：启动 VLM 推理服务使用的镜像的标签后缀。

您可以通过修改 `compose.yaml` 来满足自定义需求，例如：

<details>
<summary>1. 更改 PaddleOCR-VL 服务的端口</summary>

编辑 <code>compose.yaml</code> 文件中的 <code>paddleocr-vl-api.ports</code> 来更改端口。例如，如果您需要将服务端口更换为 8111，可以进行以下修改：

```diff
  paddleocr-vl-api:
    ...
    ports:
-     - 8080:8080
+     - 8111:8080
    ...
```

</details>

<details>
<summary>2. 指定 PaddleOCR-VL 服务所使用的 GPU</summary>

编辑 <code>compose.yaml</code> 文件中的 <code>environment</code> 来更改所使用的 GPU。例如，如果您需要使用卡 1 进行部署，可以进行以下修改：

```diff
  paddleocr-vl-api:
    ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
-             device_ids: ["0"]
+             device_ids: ["1"]
              capabilities: [gpu]
    ...
  paddleocr-vlm-server:
    ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
-             device_ids: ["0"]
+             device_ids: ["1"]
              capabilities: [gpu]
    ...
```

</details>

<details>
<summary>3. 调整 VLM 服务端配置</summary>

若您想调整 VLM 服务端的配置，可以参考 <a href="./PaddleOCR-VL.md#331-服务端参数调整">3.3.1 服务端参数调整</a> 生成配置文件。

生成配置文件后，将以下的 <code>paddleocr-vlm-server.volumes</code> 和 <code>paddleocr-vlm-server.command</code> 字段增加到您的 <code>compose.yaml</code> 中。请将 <code>/path/to/your_config.yaml</code> 替换为您的实际配置文件路径。

```yaml
  paddleocr-vlm-server:
    ...
    volumes: /path/to/your_config.yaml:/home/paddleocr/vlm_server_config.yaml
    command: paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend vllm --backend_config /home/paddleocr/vlm_server_config.yaml
    ...
```

</details>

<details>
<summary>4. 调整产线相关配置（如模型路径、批处理大小、部署设备等）</summary>

参考 <a href="./PaddleOCR-VL.md#44-产线配置调整说明">4.4 产线配置调整说明</a> 小节。

</details>

### 4.2 方法二：手动部署

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

### 4.3 客户端调用方式

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

### 4.4 产线配置调整说明

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。

## 5. 模型微调

请参考[PaddleOCR-VL 使用教程](./PaddleOCR-VL.md) 相同章节。
