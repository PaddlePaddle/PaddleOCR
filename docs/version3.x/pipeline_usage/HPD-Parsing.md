---
comments: true
---

# HPD-Parsing 使用教程

HPD-Parsing 是一款面向高吞吐文档解析的轻量级视觉语言模型。不同于传统统一式文档解析模型沿单一轨迹逐 token 串行生成，HPD-Parsing 采用层级并行解码范式：由主布局分支负责全局结构协调，并动态生成多个局部内容分支进行并发解码，同时结合渐进式多 token 预测（P-MTP）进一步减少各分支内部的解码步数。在保持具有竞争力解析精度的同时，HPD-Parsing 在公开基准上达到 4,752 tokens/s 的峰值吞吐，分别达到当前最快文档解析模型的 1.62 倍和该模型自回归基线的 3.06 倍，适用于对推理效率和部署吞吐要求较高的文档解析场景。

HPD-Parsing 基于特供版本的 vLLM 运行：该版本在 vLLM v0.17.1 的基础上实现了层级并行解码所需的动态请求分叉机制，并适配了 P-MTP 投机解码。

使用 HPD-Parsing 分为两步：先**准备运行环境**（使用官方 Docker 镜像或安装预编译包，二选一），再从以下两种**使用方式**中任选一种：

- **服务化部署**：启动 OpenAI 兼容推理服务，客户端通过 API 调用。一次部署可供多个客户端并发调用，适用于生产部署场景。
- **本地推理**：通过 vLLM Python API 在 Python 进程内直接加载模型推理，无需启动服务，适用于单机离线批量处理场景。

两种使用方式共用同一套推理引擎，均可运行在任意一种运行环境之上。

## 环境要求

- **硬件**：NVIDIA GPU（已在 H100、H800、H20、A100、A800、A30、L20、RTX Pro 6000 上完成验证），NVIDIA 驱动需支持 CUDA 12.8 或以上版本。
- **操作系统**：Linux x86-64。若使用其他操作系统，请通过 Docker 镜像的方式使用。
- **Docker 方式**：Docker 版本 >= 19.03。
- **预编译包方式**：Python 3.10–3.13。

> INFO:
> HPD-Parsing 不依赖 `paddleocr` Python 库，本教程中的服务启动、客户端调用与本地推理均不需要安装 PaddleOCR。

## 1. 准备运行环境

无论采用哪种使用方式，都需要先准备包含特供版本 vLLM 的运行环境。有以下两种方式，任选一种即可：

- 方式一：使用官方 Docker 镜像。
- 方式二：安装特供版本 vLLM 预编译包。

**我们强烈推荐采用 Docker 镜像的方式，以最大程度减少可能出现的环境问题。**

### 1.1 方式一：使用 Docker 镜像

官方 Docker 镜像内置特供版本 vLLM 及全部依赖（要求 Docker 版本 >= 19.03，机器装配有 NVIDIA GPU 且驱动支持 CUDA 12.8 或以上版本）：

```text
ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu
```

该镜像的默认入口为自动启动推理服务（用法见 2.1 节）；覆盖默认入口后也可以将其用作本地推理环境（见第 3 节）。无需提前拉取镜像，首次执行 `docker run` 时会自动拉取。

如果您希望在无法连接互联网的环境中使用 HPD-Parsing，请使用离线版本镜像 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu-offline`（镜像大小约为 24.5 GB，在线版本镜像约为 20.2 GB）。离线版本镜像内置模型权重，启动时无需连接互联网。您需要在可以联网的机器上拉取镜像，将镜像导入到离线机器。例如：

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

### 1.2 方式二：安装预编译包

如果您无法使用 Docker，也可以安装特供版本 vLLM 的预编译包。预编译包支持 Python 3.10–3.13，要求 NVIDIA 驱动支持 CUDA 12.8 或以上版本。

**我们强烈推荐您在虚拟环境中安装，以避免发生依赖冲突。** 例如，使用 Python venv 标准库创建虚拟环境：

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

> INFO:
> 除离线版本镜像已内置模型权重外，其余情况下模型权重会在首次运行时自动从 HuggingFace 下载。如果您所在网络访问 HuggingFace 较慢，可以设置环境变量 `HF_ENDPOINT=https://hf-mirror.com` 使用镜像站。

## 2. 服务化部署

服务化部署分为两步：先启动推理服务，再通过客户端调用。

### 2.1 启动服务

**使用 Docker 镜像时**，直接启动容器即可，容器将自动启动推理服务，默认监听 **8118** 端口：

```shell
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu
```

在无法连接互联网的机器上，请将上述镜像名更换为已导入的离线版本镜像。如需使用 HuggingFace 镜像站，可在命令中添加 `-e HF_ENDPOINT=https://hf-mirror.com`。

**使用预编译包时**，在安装了预编译包的环境中执行以下命令启动服务：

```shell
MAX_PATCHES_WITH_RESIZE=true vllm serve PaddlePaddle/HPD-Parsing \
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
    --speculative-config '{"method":"medusa","model":"PaddlePaddle/HPD-Parsing/P-MTP","num_speculative_tokens":6}'
```

其中的关键参数说明如下：

| 参数 | 说明 |
| --- | --- |
| `MAX_PATCHES_WITH_RESIZE=true` | 环境变量，控制图像预处理行为，必须设置。 |
| `--attention-backend FLASHINFER` | 使用 FlashInfer 注意力后端，为推荐配置。 |
| `--speculative-config` | 配置 P-MTP，`model` 指向 P-MTP 权重目录，`num_speculative_tokens` 为投机 token 数。 |
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

模型输出为结构化的文档解析结果：每个版面块以 `<BLOCK>` 开头，依次为块类别（如 `text`、`title`、`image`、`image_caption`、`header`、`page_number` 等）、位置（bbox 坐标）与 `<CHILD>` 之后的文本内容（`image` 等无文本内容的块没有 `<CHILD>` 部分）。可参考以下代码从输出中提取所有版面块：

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

HPD-Parsing 的吞吐优势在多并发场景下更为显著。批量处理文档时，建议通过多线程并发提交请求，例如：

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

除服务化方式外，也可以通过 vLLM Python API 在本地直接加载模型推理，无需启动服务，适用于单机离线批量处理场景。该方式可在安装了预编译包的 Python 环境中运行；使用 Docker 镜像时，可在 `docker run` 命令中添加 `--entrypoint /bin/bash` 覆盖默认入口，进入容器后运行脚本。

```python
# 运行前需设置环境变量：export MAX_PATCHES_WITH_RESIZE=true
import base64

from vllm import LLM, SamplingParams

llm = LLM(
    model="PaddlePaddle/HPD-Parsing",
    trust_remote_code=True,
    max_model_len=16384,
    limit_mm_per_prompt={"image": 1},
    gpu_memory_utilization=0.9,
    attention_backend="FLASHINFER",
    enable_prefix_caching=True,
    speculative_config={
        "method": "medusa",
        "model": "PaddlePaddle/HPD-Parsing/P-MTP",
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

本地推理与服务化方式共用同一套推理引擎，层级并行解码与 P-MTP 在两种方式下均会生效。批量处理时可一次性向 `llm.chat` 传入多组 `messages`，引擎将自动完成并发调度。

## 4. 性能调优

- **并发请求**：HPD-Parsing 的层级并行解码会为每个请求动态创建多个并发解码分支，吞吐优势在多并发场景下更为显著。建议客户端使用多线程/异步方式批量提交请求。
- **抑制复读**：对于版面极端复杂的文档，如遇到输出重复的情况，可在请求中设置 `"repetition_penalty": 1.05`（通过 `extra_body` 传入）缓解。
- **显存调整**：如果服务与其他程序共用 GPU，可适当调低 `--gpu-memory-utilization`；如遇长文档解析被截断，可适当调大 `--max-model-len` 与 `max_tokens`。
