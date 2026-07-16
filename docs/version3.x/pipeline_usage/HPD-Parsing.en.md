---
comments: true
---

# HPD-Parsing Usage Tutorial

HPD-Parsing is a lightweight vision-language model designed for high-throughput document parsing. Unlike traditional unified document parsing models that generate tokens serially along a single trajectory, HPD-Parsing adopts a hierarchical parallel decoding paradigm: a main layout branch coordinates the global structure while dynamically spawning multiple local content branches that decode concurrently, combined with Progressive Multi-Token Prediction (P-MTP) to further reduce the number of decoding steps within each branch. While maintaining competitive parsing accuracy, HPD-Parsing achieves a peak throughput of 4,752 tokens/s on public benchmarks, which is 1.62 times that of the fastest existing document parsing model and 3.06 times that of its autoregressive baseline, making it well suited for document parsing scenarios with high demands on inference efficiency and deployment throughput.

HPD-Parsing runs on a customized build of vLLM: based on vLLM v0.17.1, it implements the dynamic request forking mechanism required by hierarchical parallel decoding and adapts P-MTP speculative decoding.

Using HPD-Parsing takes two steps: first **prepare the runtime environment** (using the official Docker image or installing the prebuilt package, either one), then choose one of the two **usage approaches**:

- **Serving**: Start an OpenAI-compatible inference server and call it from clients via the API. One deployment can serve multiple clients concurrently, suitable for production deployment.
- **Local inference**: Load the model directly in a Python process through the vLLM Python API without starting a server, suitable for single-machine offline batch processing.

Both approaches share the same inference engine and can run on top of either runtime environment.

## Environment Requirements

- **Hardware**: NVIDIA GPU (verified on H100, H800, H20, A100, A800, A30, L20, and RTX Pro 6000), with an NVIDIA driver supporting CUDA 12.8 or later.
- **Operating system**: Linux x86-64. For other operating systems, please use the Docker image.
- **Docker approach**: Docker version >= 19.03.
- **Prebuilt package approach**: Python 3.10–3.13.

> INFO:
> HPD-Parsing does not depend on the `paddleocr` Python library; starting the server, calling it from clients, and local inference in this tutorial do not require installing PaddleOCR.

## 1. Preparing the Runtime Environment

Whichever usage approach you choose, you first need a runtime environment containing the customized build of vLLM. There are two options; either one works:

- Option 1: Use the official Docker image.
- Option 2: Install the prebuilt package of the customized vLLM build.

**We strongly recommend using the Docker image to minimize potential environment issues.**

### 1.1 Option 1: Using the Docker Image

The official Docker image ships with the customized vLLM build and all dependencies (requires Docker version >= 19.03 and a machine equipped with an NVIDIA GPU whose driver supports CUDA 12.8 or later):

```text
ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu
```

The default entrypoint of the image starts the inference server automatically (see Section 2.1); by overriding the default entrypoint, the image can also be used as a local inference environment (see Section 3). There is no need to pull the image in advance; it will be pulled automatically on the first `docker run`.

If you wish to use HPD-Parsing in an environment without internet access, use the offline image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu-offline` (image size approximately 24.5 GB; the online image is approximately 20.2 GB). The offline image has the model weights built in, so no internet connection is needed at startup. You need to pull the image on a machine with internet access and transfer it to the offline machine. For example:

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

### 1.2 Option 2: Installing the Prebuilt Package

If you cannot use Docker, you can also install the prebuilt package of the customized vLLM build. The prebuilt package supports Python 3.10–3.13 and requires an NVIDIA driver supporting CUDA 12.8 or later.

**We strongly recommend installing in a virtual environment to avoid dependency conflicts.** For example, create one with the Python venv standard library:

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

> INFO:
> Except for the offline image, which has the model weights built in, the model weights are downloaded automatically from HuggingFace on first run. If your network has slow access to HuggingFace, set the environment variable `HF_ENDPOINT=https://hf-mirror.com` to use a mirror site.

## 2. Serving

Serving involves two steps: start the inference server, then call it from a client.

### 2.1 Starting the Server

**When using the Docker image**, simply start the container; it will start the inference server automatically, listening on port **8118** by default:

```shell
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm:latest-nvidia-gpu
```

On a machine without internet access, replace the image name above with the imported offline image. To use a HuggingFace mirror site, add `-e HF_ENDPOINT=https://hf-mirror.com` to the command.

**When using the prebuilt package**, run the following command in the environment where the package is installed:

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

Key parameters are explained below:

| Parameter | Description |
| --- | --- |
| `MAX_PATCHES_WITH_RESIZE=true` | Environment variable controlling image preprocessing behavior; must be set. |
| `--attention-backend FLASHINFER` | Use the FlashInfer attention backend; recommended configuration. |
| `--speculative-config` | Configures P-MTP; `model` points to the P-MTP weights directory, and `num_speculative_tokens` is the number of speculative tokens. |
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

The model outputs structured document parsing results: each layout block starts with `<BLOCK>`, followed by the block category (such as `text`, `title`, `image`, `image_caption`, `header`, `page_number`), the position (bbox coordinates), and the text content after `<CHILD>` (blocks without text content, such as `image`, have no `<CHILD>` part). The following code shows how to extract all layout blocks from the output:

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

The throughput advantage of HPD-Parsing is more pronounced under high concurrency. When processing documents in batches, we recommend submitting requests concurrently with multiple threads, for example:

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

In addition to serving, you can also load the model and run inference locally through the vLLM Python API without starting a server, which is suitable for single-machine offline batch processing. This approach can run in a Python environment with the prebuilt package installed; when using the Docker image, add `--entrypoint /bin/bash` to the `docker run` command to override the default entrypoint, then run the script inside the container.

```python
# Set the environment variable before running: export MAX_PATCHES_WITH_RESIZE=true
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

Local inference shares the same engine as the serving approach; hierarchical parallel decoding and P-MTP take effect in both. For batch processing, you can pass multiple `messages` to `llm.chat` at once, and the engine will handle concurrent scheduling automatically.

## 4. Performance Tuning

- **Concurrent requests**: Hierarchical parallel decoding dynamically creates multiple concurrent decoding branches for each request, and the throughput advantage is more pronounced under high concurrency. We recommend submitting requests in batches from the client using multithreading or asynchronous calls.
- **Suppressing repetition**: For documents with extremely complex layouts, if repeated output is observed, setting `"repetition_penalty": 1.05` in the request (via `extra_body`) can mitigate it.
- **GPU memory adjustment**: If the server shares the GPU with other programs, lower `--gpu-memory-utilization` appropriately; if long-document parsing results are truncated, increase `--max-model-len` and `max_tokens`.
