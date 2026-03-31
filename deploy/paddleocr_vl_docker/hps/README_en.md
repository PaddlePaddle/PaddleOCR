# PaddleOCR-VL-1.5 High-Performance Service Deployment

[简体中文](README.md)

This directory provides a high-performance service deployment solution for PaddleOCR-VL-1.5 with concurrent request processing support.

> This solution currently only supports NVIDIA GPUs. Support for other inference devices is still being developed.

## Architecture

```
Client → FastAPI Gateway → Triton Server → vLLM Server
```

| Component       | Description                                                           |
|-----------------|-----------------------------------------------------------------------|
| FastAPI Gateway | Unified access point, simplified client calls, concurrency control    |
| Triton Server   | Layout detection model (PP-DocLayoutV3) and pipeline orchestration; model management, dynamic batching, inference scheduling |
| vLLM Server     | VLM (PaddleOCR-VL-1.5), continuous batching inference                |

**Triton Models:**

| Model | Device | Description |
|-------|--------|-------------|
| `layout-parsing` | Inference device (e.g., GPU) | Layout parsing inference |
| `restructure-pages` | CPU | Multi-page result post-processing (cross-page table merging, title level reassignment) |

## Requirements

- x64 CPU
- NVIDIA GPU, Compute Capability >= 8.0 and < 12.0
- NVIDIA driver supporting CUDA 12.6
- Docker >= 19.03
- Docker Compose >= 2.0

## Quick Start

1. Clone the PaddleOCR repository and navigate to this directory:

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/deploy/paddleocr_vl_docker/hps
```

2. Prepare necessary files:

```bash
bash prepare.sh
```

3. Start the services:

```bash
docker compose up
```

The above command will start 3 containers in sequence:

| Service | Description | Port |
|---------|-------------|------|
| `paddleocr-vl-api` | FastAPI gateway (external entry point) | 8080 |
| `paddleocr-vl-tritonserver` | Triton inference server | 8000 (internal) |
| `paddleocr-vlm-server` | vLLM-based VLM inference service | 8080 (internal) |

> The first startup will automatically download and build images, which takes longer. Subsequent startups will use local images and start faster.

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and modify as needed.

```bash
cp .env.example .env
```

You can also set these as environment variables directly instead of using the `.env` file, e.g.:

```bash
export HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=8
```

| Variable | Default | Description |
|----------|---------|-------------|
| `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS` | 16 | Max concurrent inference requests (layout parsing) |
| `HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS` | 64 | Max concurrent non-inference requests (page restructuring) |
| `HPS_INFERENCE_TIMEOUT` | 600 | Request timeout in seconds |
| `HPS_HEALTH_CHECK_TIMEOUT` | 5 | Health check timeout in seconds |
| `HPS_VLM_URL` | http://paddleocr-vlm-server:8080 | VLM server URL (for health checks) |
| `HPS_LOG_LEVEL` | INFO | Log level (DEBUG, INFO, WARNING, ERROR) |
| `HPS_FILTER_HEALTH_ACCESS_LOG` | true | Whether to filter health check access logs |
| `UVICORN_WORKERS` | 4 | Number of gateway worker processes |
| `DEVICE_ID` | 0 | Inference device ID to use |

### Pipeline Configuration

To adjust pipeline configurations (such as model path, batch size, deployment device, etc.), please refer to the Pipeline Configuration section in the [PaddleOCR-VL Usage Tutorial](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md).

## API Usage

### Document Parsing

Please refer to the Client-Side Invocation section in the [PaddleOCR-VL Usage Tutorial](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md).

### Health Checks

```bash
# Liveness check
curl http://localhost:8080/health

# Readiness check (verifies Triton and VLM services are ready to process requests)
curl http://localhost:8080/health/ready
```

## Performance Tuning

### Concurrency Settings

The gateway controls concurrency for inference and non-inference operations independently:

- **`HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`** (default 16): Controls concurrency for inference operations such as `layout-parsing` (layout parsing)
  - Too low (4): Underutilized inference device, requests queue unnecessarily
  - Too high (64): May overload Triton, causing OOM or timeouts
  - Default value of 16 allows enough requests to queue for the next batch while the current batch is being processed
  - If inference device resources are limited, consider lowering this value
- **`HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS`** (default 64): Controls concurrency for non-inference operations such as `restructure-pages` (page restructuring)
  - Non-inference operations do not consume inference device resources and can be set to a higher concurrency level
  - Adjust based on CPU cores and available memory

**High-throughput configuration example:**

```bash
# .env
HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=32
HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS=128
UVICORN_WORKERS=8
```

**Low-latency configuration example:**

```bash
# .env
HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=8
HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS=32
HPS_INFERENCE_TIMEOUT=300
UVICORN_WORKERS=2
```

### Worker Processes

Each Uvicorn worker is an independent process with its own event loop:

- **1 worker**: Simple, but limited to a single process
- **4 workers**: Suitable for most scenarios
- **8+ workers**: Suitable for high-concurrency scenarios with many small requests

### Triton Dynamic Batching

Triton automatically batches requests to improve inference device utilization. The maximum batch size is controlled by the `max_batch_size` parameter in the model configuration file (default: 8), located at `config.pbtxt` under each model directory in the model repository (e.g., `model_repo/layout-parsing/config.pbtxt`).

### Triton Instance Count

The number of parallel inference instances for each Triton model is configured via the `instance_group` section in `config.pbtxt` (default: 1). Increasing the instance count improves parallelism but consumes more device resources.

```
# model_repo/layout-parsing/config.pbtxt
instance_group [
  {
      count: 1       # Number of instances; increase for higher parallelism
      kind: KIND_GPU
      gpus: [ 0 ]
  }
]
```

There is a trade-off between instance count and dynamic batching:

- **Single instance (`count: 1`)**: Dynamic batching combines multiple requests into one batch for parallel execution, but all requests in the same batch must wait for the slowest one to finish before results are returned, which may increase latency for faster requests. Additionally, a single instance can only process one batch at a time — subsequent requests must queue until the current batch completes. Best suited for scenarios with limited GPU memory or uniform request processing times
- **Multiple instances (`count: 2+`)**: Multiple instances can process different batches simultaneously, allowing more requests to be handled concurrently. This reduces queuing time and improves latency for individual requests. Note that within each instance, dynamic batching behavior still applies (requests in the same batch start and finish together). Each additional instance consumes an extra copy of the layout detection model's GPU memory, increases the load on the VLM inference service, and uses more CPU and system memory. Adjust based on the available resources of your inference device

Non-inference models (e.g., `restructure-pages`) run on CPU and can have their instance count increased based on available CPU cores.

## Troubleshooting and Resolution

### Service Fails to Start

Check the logs for each service to identify the issue:

```bash
docker compose logs paddleocr-vl-api
docker compose logs paddleocr-vl-tritonserver
docker compose logs paddleocr-vlm-server
```

Common causes include port conflicts, unavailable inference devices, or image pull failures.

### Timeout Errors

- Increase `HPS_INFERENCE_TIMEOUT` (for complex documents)
- If the inference device is overloaded, reduce `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`

### Out of Memory

- Reduce `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`
- Ensure only one service runs per inference device
- Check `shm_size` in compose.yaml (default: 4GB)
