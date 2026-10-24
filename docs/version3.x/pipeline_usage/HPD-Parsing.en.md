---
comments: true
---

# HPD-Parsing Usage Tutorial

HPD-Parsing is a lightweight vision-language model designed for high-throughput document parsing. Unlike traditional unified document parsing models that generate tokens serially along a single trajectory, HPD-Parsing adopts a hierarchical parallel decoding paradigm: a main layout branch coordinates the global structure while dynamically spawning multiple local content branches that decode concurrently, combined with Progressive Multi-Token Prediction (P-MTP) to further reduce the number of decoding steps within each branch. While maintaining competitive parsing accuracy, HPD-Parsing achieves a peak throughput of 4,752 tokens/s on public benchmarks, which is 1.62 times that of the fastest existing document parsing model and 3.06 times that of its autoregressive baseline, making it well suited for document parsing scenarios with high demands on inference efficiency and deployment throughput.

HPD-Parsing requires a customized build of vLLM. Based on vLLM v0.17.1, this build adds the dynamic request-forking mechanism required for hierarchical parallel decoding and supports P-MTP speculative decoding.

Using HPD-Parsing involves two stages: first, **prepare the runtime environment** with either the official Docker image or the prebuilt package plus downloaded model weights; then choose one of the following **inference modes**:

- **Serving**: Start an OpenAI-compatible inference server and call it from clients via the API. A single server can handle requests from multiple clients concurrently, making this mode suitable for production.
- **Local inference**: Load the model directly in the current Python process through the vLLM Python API, without starting a server. This mode is suitable for single-machine batch processing.

Both inference modes use the same engine and support either runtime environment.

## Environment Requirements

- **Hardware**: NVIDIA GPU (verified on H100, H800, H20, A100, A800, A30, L20, and RTX Pro 6000), with an NVIDIA driver supporting CUDA 12.8 or later.
- **Operating system**: Linux x86-64. Other operating systems are supported only through a Docker environment capable of running Linux NVIDIA GPU containers.
- **Docker image**: Docker >= 19.03 with NVIDIA Container Toolkit installed.
- **Prebuilt package**: Python 3.10–3.13.

> INFO:
> HPD-Parsing does not depend on the `paddleocr` Python library; starting the server, calling it from clients, and local inference in this tutorial do not require installing PaddleOCR.

## 1. Preparing the Runtime Environment

Both serving and local inference require the customized vLLM runtime. Choose either of the following options:

- Option 1: Use the official Docker image.
- Option 2: Install the customized vLLM prebuilt package and download the model weights.

**We strongly recommend using the Docker image to minimize potential environment issues.**

### 1.1 Option 1: Using the Docker Image

The official Docker image includes the customized vLLM build and all of its dependencies:

```text
ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu
```

By default, the image starts the inference server (see Section 2.1). You can also override its entrypoint and use it as a local inference environment (see Section 3). You do not need to run `docker pull` first; Docker pulls the image automatically on the first `docker run`.

To run HPD-Parsing without internet access, use the offline image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu-offline`. The offline image is approximately 24.5 GB and includes the model weights, so startup requires no network access; the online image is approximately 20.2 GB. First pull and export the offline image on a networked machine, then transfer and import it on the offline machine:

```shell
# Run on a machine with internet access
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu-offline
# Save the image to a file
docker save ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu-offline -o hpd-parsing-vllm-latest-nvidia-gpu-offline.tar

# Transfer the image file to the offline machine

# Run on the offline machine
docker load -i hpd-parsing-vllm-latest-nvidia-gpu-offline.tar
```

> TIP:
> Images tagged with `latest-xxx` correspond to the latest version.
> If a corresponding `latest` image already exists locally but you want the latest features or fixes, we recommend re-running `docker pull` to update the image before continuing.

### 1.2 Option 2: Installing the Prebuilt Package and Preparing Model Weights

If Docker is unavailable, install the customized vLLM prebuilt package. It supports Python 3.10–3.13 and requires an NVIDIA driver that supports CUDA 12.8 or later.

**Install the package in a virtual environment to avoid dependency conflicts.** For example, create one with Python's standard-library `venv` module:

```shell
# Create a virtual environment
python -m venv .venv_hpd_parsing
# Activate the environment
source .venv_hpd_parsing/bin/activate
```

Run the following command to install:

```shell
python -m pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/hpd_parsing/vllm-0.17.1+hpdparsing-cp38-abi3-manylinux_2_31_x86_64.whl
```

After installation, download the complete model repository, which contains both the main model and the `P-MTP` speculative decoding model:

```shell
hf download PaddlePaddle/HPD-Parsing --local-dir ./HPD-Parsing
```

After the download, verify that both `./HPD-Parsing/config.json` and `./HPD-Parsing/P-MTP/config.json` exist, and keep the directory structure intact. The serving instructions in Section 2 and local inference instructions in Section 3 both use this directory. For an offline machine, use `pip download` on a networked machine with the same Python version and operating-system architecture to download the prebuilt package and all of its dependencies. Download the complete model directory as well, then transfer these files to the offline machine for installation.

> INFO:
> If your network has slow access to Hugging Face, set `HF_ENDPOINT=https://hf-mirror.com` before downloading the model or starting the online Docker image.

## 2. Serving

Serving involves two steps: start the inference server, then call it from a client.

### 2.1 Starting the Server

**With the online Docker image**, start the container directly. It downloads the model and starts the inference server, which listens on port **8118** by default:

```shell
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu
```

By default, the model is cached inside the container and the cache is deleted with the container. To reuse the model cache in subsequent containers, add the optional parameter `-v hpd_parsing_hf_cache:/home/hpd/.cache/huggingface` to the `docker run` command. To use a Hugging Face mirror, add `-e HF_ENDPOINT=https://hf-mirror.com`.

**With the offline Docker image**, run the imported offline image directly. The model weights are built in, so no network connection or cache volume is required:

```shell
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu-offline
```

**With the prebuilt package**, activate the virtual environment created in Section 1.2, change to the directory containing the `HPD-Parsing` model directory, and run:

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

Key parameters are explained below:

| Parameter | Description |
| --- | --- |
| `MAX_PATCHES_WITH_RESIZE=true` | Environment variable controlling image preprocessing behavior; must be set. |
| `--attention-backend FLASHINFER` | Use the FlashInfer attention backend; recommended configuration. |
| `--speculative-config` | Configures P-MTP. `model` points to the P-MTP weights directory, and `num_speculative_tokens` sets the number of speculative tokens generated per step. |
| `--max-model-len` | Maximum context length; adjust according to available GPU memory. |
| `--gpu-memory-utilization` | Fraction of GPU memory to use; adjust as needed. |

### 2.2 Client Usage

Once the server is running, it can be called through the OpenAI-compatible API. When parsing document images, the prompt is fixed as `document parsing with fork.`. The server automatically performs hierarchical parallel decoding, and a single request returns the complete parsing result.

Here is an example of calling the server with the `openai` Python client library. Install the client library first:

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

The model returns a structured document representation. Each layout block starts with `<BLOCK>`, followed by its category (such as `text`, `title`, `image`, `image_caption`, `header`, or `page_number`), bounding-box coordinates, and text content after `<CHILD>`. Blocks without text content, such as `image`, do not include `<CHILD>`. The following function extracts all layout blocks:

```python
import re

def parse_blocks(input_text: str) -> list[dict]:
    """Parse all layout blocks from the output"""
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

HPD-Parsing provides its greatest throughput benefit when multiple requests run concurrently. For batch processing, submit requests concurrently with threads or asynchronous I/O. The following example uses a thread pool to process multiple images:

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

## 3. Local Inference (Python API)

As an alternative to serving, you can load the model directly in the current process through the vLLM Python API. This requires no inference server and is suitable for single-machine batch processing. First, save the following code as `hpd_infer.py`, and place the input image at `demo.png` in the same directory:

```python
import base64
import os
from pathlib import Path

from vllm import LLM, SamplingParams

model_path = Path(os.environ["MODEL_PATH"])
if not (model_path / "config.json").is_file():
    raise FileNotFoundError(f"Invalid main model directory: {model_path}")
if not (model_path / "P-MTP" / "config.json").is_file():
    raise FileNotFoundError(f"Invalid P-MTP model directory: {model_path / 'P-MTP'}")

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

**When using the online Docker image**, run the following command from the directory containing `hpd_infer.py` and `demo.png`:

```shell
docker run --rm --gpus all \
    --entrypoint /bin/bash \
    -v "$(pwd):/workspace" \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu \
    -lc 'export MODEL_PATH="$(hf download PaddlePaddle/HPD-Parsing --quiet)"; cd /workspace; python hpd_infer.py'
```

The online image downloads the model on first use. To reuse the downloaded model in subsequent containers, add the optional parameter `-v hpd_parsing_hf_cache:/home/hpd/.cache/huggingface` to the `docker run` command. To use a Hugging Face mirror, add `-e HF_ENDPOINT=https://hf-mirror.com`.

**When using the offline Docker image**, the model weights are already built in, so no network connection or model cache volume is required:

```shell
docker run --rm --gpus all \
    --entrypoint /bin/bash \
    -e MODEL_PATH=/home/hpd/models/HPD-Parsing \
    -v "$(pwd):/workspace" \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu-offline \
    -lc 'cd /workspace; python hpd_infer.py'
```

**With the prebuilt package**, ensure that the current directory contains `hpd_infer.py`, `demo.png`, and the `HPD-Parsing` directory downloaded in Section 1.2, then run:

```shell
MAX_PATCHES_WITH_RESIZE=true \
MODEL_PATH="$(realpath ./HPD-Parsing)" \
python hpd_infer.py
```

All three approaches use the same inference engine, with hierarchical parallel decoding and P-MTP enabled. For batch processing, pass a list of conversations to `llm.chat`; the engine schedules the batch automatically.

## 4. Performance Tuning

- **Concurrent requests**: Hierarchical parallel decoding dynamically creates multiple concurrent decoding branches for each request, and the throughput advantage is more pronounced under high concurrency. We recommend submitting requests in batches from the client using multithreading or asynchronous calls.
- **Reduce repeated output**: When parsing documents with extremely complex layouts, set `extra_body={"repetition_penalty": 1.05}` in the client request if the model repeats content.
- **Control GPU memory use**: When sharing a GPU with other processes, lower `--gpu-memory-utilization` for serving or the `LLM` parameter `gpu_memory_utilization` for local inference.
- **Increase output length**: If a result is truncated because it reaches the output limit, increase `max_tokens` in the client request or `SamplingParams`. If the combined input and expected output exceed the context limit, also increase `--max-model-len` for serving or the `LLM` parameter `max_model_len` for local inference. Larger values consume more GPU memory.
