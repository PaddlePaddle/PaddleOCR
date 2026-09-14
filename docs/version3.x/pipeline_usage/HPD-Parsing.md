---
comments: true
---

# HPD-Parsing 使用教程

HPD-Parsing 是一款面向高吞吐文档解析的轻量级视觉语言模型。不同于传统统一式文档解析模型沿单一轨迹逐 token 串行生成，HPD-Parsing 采用层级并行解码范式：由主布局分支负责全局结构协调，并动态生成多个局部内容分支进行并发解码，同时结合渐进式多 token 预测（P-MTP）进一步减少各分支内部的解码步数。在保持具有竞争力解析精度的同时，HPD-Parsing 在公开基准上达到 4,752 tokens/s 的峰值吞吐，分别达到当前最快文档解析模型的 1.62 倍和该模型自回归基线的 3.06 倍，适用于对推理效率和部署吞吐要求较高的文档解析场景。

HPD-Parsing 需要使用定制版 vLLM。该版本基于 vLLM v0.17.1，增加了层级并行解码所需的动态请求分叉机制，并支持 P-MTP 投机解码。

使用 HPD-Parsing 分为两步：先选择一种方式**准备运行环境**（使用官方 Docker 镜像，或安装预编译包并下载模型权重），再选择一种**推理方式**：

- **服务化部署**：启动 OpenAI 兼容推理服务，客户端通过 API 调用。一次部署可供多个客户端并发调用，适用于生产部署场景。
- **本地推理**：通过 vLLM Python API 在当前 Python 进程中直接加载模型，无需启动服务，适用于单机批量处理。

两种推理方式使用同一套推理引擎，并且都支持 Docker 镜像和预编译包环境。

## 环境要求

- **硬件**：NVIDIA GPU（已在 H100、H800、H20、A100、A800、A30、L20、RTX Pro 6000 上完成验证），NVIDIA 驱动需支持 CUDA 12.8 或以上版本。
- **操作系统**：Linux x86-64。其他操作系统仅支持通过能够运行 Linux NVIDIA GPU 容器的 Docker 环境使用。
- **Docker 镜像方式**：Docker 版本 >= 19.03，并已安装 NVIDIA Container Toolkit。
- **预编译包方式**：Python 3.10–3.13。

> INFO:
> HPD-Parsing 不依赖 `paddleocr` Python 库，本教程中的服务启动、客户端调用与本地推理均不需要安装 PaddleOCR。

## 1. 准备运行环境

无论采用服务化部署还是本地推理，都需要先准备定制版 vLLM 运行环境。以下两种方式任选其一：

- 方式一：使用官方 Docker 镜像。
- 方式二：安装定制版 vLLM 预编译包并下载模型权重。

**我们强烈推荐采用 Docker 镜像的方式，以最大程度减少可能出现的环境问题。**

### 1.1 方式一：使用 Docker 镜像

官方 Docker 镜像内置定制版 vLLM 及其全部依赖：

```text
ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu
```

该镜像默认启动推理服务（见 2.1 节）。也可以覆盖默认入口，将其用作本地推理环境（见第 3 节）。无需提前执行 `docker pull`；首次执行 `docker run` 时，Docker 会自动拉取镜像。

如需在无法连接互联网的环境中运行 HPD-Parsing，请使用离线镜像 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu-offline`。离线镜像约为 24.5 GB，内置模型权重，启动时无需联网；在线镜像约为 20.2 GB。请先在联网机器上拉取并导出离线镜像，再将其传输到离线机器并导入。例如：

```shell
# 在能够联网的机器上执行
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu-offline
# 将镜像保存到文件中
docker save ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu-offline -o hpd-parsing-vllm-latest-nvidia-gpu-offline.tar

# 将镜像文件传输到离线机器

# 在离线机器上执行
docker load -i hpd-parsing-vllm-latest-nvidia-gpu-offline.tar
```

> TIP:
> 标签后缀为 `latest-xxx` 的镜像对应最新版本。
> 如果本地已经存在对应的 `latest` 镜像，但希望使用最新功能或修复，建议在继续使用前重新执行一次 `docker pull` 更新镜像。

### 1.2 方式二：安装预编译包并准备模型权重

如果无法使用 Docker，可以安装定制版 vLLM 的预编译包。该预编译包支持 Python 3.10–3.13，并要求 NVIDIA 驱动支持 CUDA 12.8 或以上版本。

**建议在虚拟环境中安装，以避免依赖冲突。** 例如，可以使用 Python 标准库中的 `venv` 创建虚拟环境：

```shell
# 创建虚拟环境
python -m venv .venv_hpd_parsing
# 激活环境
source .venv_hpd_parsing/bin/activate
```

执行如下命令完成安装：

```shell
python -m pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/hpd_parsing/vllm-0.17.1+hpdparsing-cp38-abi3-manylinux_2_31_x86_64.whl
```

安装完成后，下载完整模型仓库。该仓库同时包含主模型和 `P-MTP` 投机解码模型：

```shell
hf download PaddlePaddle/HPD-Parsing --local-dir ./HPD-Parsing
```

下载完成后，请确认 `./HPD-Parsing/config.json` 和 `./HPD-Parsing/P-MTP/config.json` 均存在，并保留完整目录结构。第 2 节的服务化部署和第 3 节的本地推理都会使用该目录。如需在离线机器上使用预编译包，请在具有相同 Python 版本和操作系统架构的联网机器上，通过 `pip download` 下载预编译包及其全部依赖，同时下载完整模型目录，再将这些文件传输到离线机器安装。

> INFO:
> 如果您所在网络访问 Hugging Face 较慢，可以在下载模型或启动在线 Docker 镜像前设置环境变量 `HF_ENDPOINT=https://hf-mirror.com` 使用镜像站。

## 2. 服务化部署

服务化部署分为两步：先启动推理服务，再通过客户端调用。

### 2.1 启动服务

**使用在线 Docker 镜像时**，直接启动容器即可。容器会自动下载模型并启动推理服务，默认监听 **8118** 端口：

```shell
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu
```

模型默认缓存在容器内，容器删除后缓存也会丢失。如需在后续容器中复用模型缓存，可在 `docker run` 命令中添加可选参数 `-v hpd_parsing_hf_cache:/home/hpd/.cache/huggingface`。如需使用 Hugging Face 镜像站，可添加 `-e HF_ENDPOINT=https://hf-mirror.com`。

**使用离线 Docker 镜像时**，直接运行已导入的离线镜像。模型权重已内置，无需联网或挂载缓存卷：

```shell
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu-offline
```

**使用预编译包时**，请先激活第 1.2 节创建的虚拟环境，再进入 `HPD-Parsing` 模型目录所在的目录并执行以下命令：

```shell
MODEL_PATH="$(realpath ./HPD-Parsing)"
MAX_PATCHES_WITH_RESIZE=true vllm serve "${MODEL_PATH}" \
    --trust-remote-code \
    --port 8118 \
    --served-model-name HPD-Parsing \
    --max-model-len 16384 \
    --limit-mm-per-prompt '{"image": 1}' \
    --gpu-memory-utilization 0.9 \
    --attention-backend FLASHINFER \
    --attention-config '{"use_prefill_query_quantization":true}' \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --speculative-config "{\"method\":\"medusa\",\"model\":\"${MODEL_PATH}/P-MTP\",\"num_speculative_tokens\":6}"
```

其中的关键参数说明如下：

| 参数 | 说明 |
| --- | --- |
| `MAX_PATCHES_WITH_RESIZE=true` | 环境变量，控制图像预处理行为，必须设置。 |
| `--attention-backend FLASHINFER` | 使用 FlashInfer 注意力后端，为推荐配置。 |
| `--speculative-config` | 配置 P-MTP；`model` 指向 P-MTP 权重目录，`num_speculative_tokens` 指定每步生成的投机 token 数量。 |
| `--max-model-len` | 最大上下文长度，可根据显存情况调整。 |
| `--gpu-memory-utilization` | 显存占用比例，可根据实际情况调整。 |

### 2.2 客户端调用

服务启动后，可通过 OpenAI 兼容 API 调用。解析文档图像时，提示词固定为 `document parsing with fork.`，服务端将自动完成层级并行解码，一次请求即可返回完整解析结果。

以下是使用 `openai` Python 客户端库调用服务的示例。请先安装客户端库：

```shell
python -m pip install openai
```

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8118/v1", api_key="EMPTY")

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_base64 = encode_image("demo.png")

response = client.chat.completions.create(
    model="HPD-Parsing",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
                {"type": "text", "text": "document parsing with fork."},
            ],
        }
    ],
    max_tokens=8000,
    temperature=0,
)
print(response.choices[0].message.content)
```

模型返回结构化的文档解析结果。每个版面块以 `<BLOCK>` 开头，后面依次是块类别（如 `text`、`title`、`image`、`image_caption`、`header`、`page_number` 等）、边界框坐标，以及 `<CHILD>` 后的文本内容。`image` 等无文本内容的块不包含 `<CHILD>`。以下代码可从输出中提取所有版面块：

```python
import re

def parse_blocks(input_text: str) -> list[dict]:
    """解析输出中的所有版面块"""
    pattern = re.compile(
        r"<BLOCK>(\w+)\s*\[([^\]]*)\](?:<CHILD>)?(.*?)(?=<BLOCK>|\Z)", re.DOTALL
    )
    blocks = []
    for block_type, coords_str, content in pattern.findall(input_text):
        blocks.append(
            {
                "type": block_type,
                "bbox": [int(x.strip()) for x in coords_str.split(",")],
                "text": content.strip(),
            }
        )
    return blocks
```

HPD-Parsing 的吞吐优势在多请求并发场景下更为明显。批量处理文档时，建议使用多线程或异步方式并发提交请求。以下示例使用线程池处理多张图像：

```python
import base64
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8118/v1", api_key="EMPTY")

def parse_one(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    response = client.chat.completions.create(
        model="HPD-Parsing",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                    {"type": "text", "text": "document parsing with fork."},
                ],
            }
        ],
        max_tokens=8000,
        temperature=0,
    )
    return response.choices[0].message.content

image_paths = ["page_1.png", "page_2.png", "page_3.png"]
with ThreadPoolExecutor(max_workers=16) as executor:
    results = list(executor.map(parse_one, image_paths))

for path, result in zip(image_paths, results):
    print(path, len(result))
```

## 3. 本地推理（Python API）

除了启动推理服务，还可以通过 vLLM Python API 在当前进程中直接加载模型。这种方式无需启动服务，适用于单机批量处理。首先，将以下代码保存为 `hpd_infer.py`，并将待解析图像保存为同一目录下的 `demo.png`：

```python
import base64
import os
from pathlib import Path

from vllm import LLM, SamplingParams

model_path = Path(os.environ["MODEL_PATH"])
if not (model_path / "config.json").is_file():
    raise FileNotFoundError(f"主模型目录无效：{model_path}")
if not (model_path / "P-MTP" / "config.json").is_file():
    raise FileNotFoundError(f"P-MTP 模型目录无效：{model_path / 'P-MTP'}")

llm = LLM(
    model=str(model_path),
    trust_remote_code=True,
    max_model_len=16384,
    limit_mm_per_prompt={"image": 1},
    gpu_memory_utilization=0.9,
    attention_backend="FLASHINFER",
    enable_prefix_caching=True,
    speculative_config={
        "method": "medusa",
        "model": str(model_path / "P-MTP"),
        "num_speculative_tokens": 6,
    },
)
sampling_params = SamplingParams(temperature=0, max_tokens=8000)

with open("demo.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            },
            {"type": "text", "text": "document parsing with fork."},
        ],
    }
]

outputs = llm.chat(messages=messages, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

**使用在线 Docker 镜像时**，在包含 `hpd_infer.py` 和 `demo.png` 的目录中执行：

```shell
docker run --rm --gpus all \
    --entrypoint /bin/bash \
    -v "$(pwd):/workspace" \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu \
    -lc 'export MODEL_PATH="$(hf download PaddlePaddle/HPD-Parsing --quiet)"; cd /workspace; python hpd_infer.py'
```

在线镜像首次运行时需要下载模型。如需在后续容器中复用下载结果，可在 `docker run` 命令中添加可选参数 `-v hpd_parsing_hf_cache:/home/hpd/.cache/huggingface`。如需使用 Hugging Face 镜像站，可添加 `-e HF_ENDPOINT=https://hf-mirror.com`。

**使用离线 Docker 镜像时**，模型权重已内置，无需联网或挂载模型缓存：

```shell
docker run --rm --gpus all \
    --entrypoint /bin/bash \
    -e MODEL_PATH=/home/hpd/models/HPD-Parsing \
    -v "$(pwd):/workspace" \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu-offline \
    -lc 'cd /workspace; python hpd_infer.py'
```

**使用预编译包时**，请确认当前目录同时包含 `hpd_infer.py`、`demo.png` 和第 1.2 节下载的 `HPD-Parsing` 目录，然后执行：

```shell
MAX_PATCHES_WITH_RESIZE=true \
MODEL_PATH="$(realpath ./HPD-Parsing)" \
python hpd_infer.py
```

以上三种运行方式共用同一套推理引擎，层级并行解码与 P-MTP 均会生效。批量处理时，可将多组对话以列表形式传给 `llm.chat`，引擎将自动完成批量调度。

## 4. 性能调优

- **并发请求**：HPD-Parsing 的层级并行解码会为每个请求动态创建多个并发解码分支，吞吐优势在多并发场景下更为显著。建议客户端使用多线程/异步方式批量提交请求。
- **减少重复输出**：解析版面极其复杂的文档时，如果模型输出重复内容，可以在客户端请求中设置 `extra_body={"repetition_penalty": 1.05}`。
- **调整显存占用**：如果与其他程序共用 GPU，服务化部署时可调低 `--gpu-memory-utilization`，本地推理时可调低 `LLM` 的 `gpu_memory_utilization`。
- **调整输出长度**：如果解析结果因达到输出上限而被截断，可在客户端请求或 `SamplingParams` 中调大 `max_tokens`。如果输入与期望输出的总长度超过上下文限制，服务化部署时还需调大 `--max-model-len`，本地推理时则需调大 `LLM` 的 `max_model_len`。增大这些值会占用更多显存。
