# PaddleOCR-VL 高性能服务化部署

[English](README_en.md)

本目录提供一套支持并发请求处理的 PaddleOCR-VL 高性能服务化部署方案。

> 本方案目前暂时只支持 NVIDIA GPU，对其他推理设备的支持仍在完善中。

## 架构

```
客户端 → FastAPI 网关 → Triton 服务器 → vLLM 服务器
```

| 组件           | 说明                                   |
|----------------|----------------------------------------|
| FastAPI 网关   | 统一访问入口、简化客户端调用、并发控制 |
| Triton 服务器  | 模型管理、动态批处理、推理调度         |
| vLLM 服务器    | 连续批处理、VLM 推理                   |

**Triton 模型：**

| 模型 | 设备 | 说明 |
|------|------|------|
| `layout-parsing` | 推理设备（如 GPU） | 版面解析推理 |
| `restructure-pages` | CPU | 多页结果后处理（跨页表格合并、标题层级重分配） |

## 环境要求

- x64 CPU
- NVIDIA GPU，Compute Capability >= 8.0 且 < 12.0
- NVIDIA 驱动支持 CUDA 12.6
- Docker >= 19.03
- Docker Compose >= 2.0

## 快速开始

1. 拉取 PaddleOCR 源码并切换到当前目录：

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/deploy/paddleocr_vl_docker/hps
```

2. 准备必要文件：

```bash
bash prepare.sh
```

3. 启动服务：

```bash
docker compose up
```

上述命令将依次启动 3 个容器：

| 服务 | 说明 | 端口 |
|------|------|------|
| `paddleocr-vl-api` | FastAPI 网关（对外入口） | 8080 |
| `paddleocr-vl-tritonserver` | Triton 推理服务器 | 8000（内部） |
| `paddleocr-vlm-server` | 基于 vLLM 的 VLM 推理服务 | 8080（内部） |

> 首次启动会自动下载并构建镜像，耗时较长；从第二次启动起将直接使用本地镜像，启动速度更快。

## 配置说明

### 环境变量

复制 `.env.example` 到 `.env` 并根据需要修改。

```bash
cp .env.example .env
```

除了通过 `.env` 文件设置，也可以直接设置环境变量，如：

```bash
export HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=8
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS` | 16 | 推理操作（版面解析）最大并发请求数 |
| `HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS` | 64 | 非推理操作（多页重组）最大并发请求数 |
| `HPS_INFERENCE_TIMEOUT` | 600 | 请求超时时间（秒） |
| `HPS_HEALTH_CHECK_TIMEOUT` | 5 | 健康检查超时时间（秒） |
| `HPS_VLM_URL` | http://paddleocr-vlm-server:8080 | VLM 服务器地址（用于健康检查） |
| `HPS_LOG_LEVEL` | INFO | 日志级别（DEBUG, INFO, WARNING, ERROR） |
| `HPS_FILTER_HEALTH_ACCESS_LOG` | true | 是否过滤健康检查的访问日志 |
| `UVICORN_WORKERS` | 4 | 网关 Worker 进程数 |
| `DEVICE_ID` | 0 | 使用的推理设备 ID |

### 产线配置调整

如需调整产线相关配置（如模型路径、批处理大小、部署设备等），请参考 [PaddleOCR-VL 使用教程](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.md) 中的产线配置调整说明章节。

## API 使用

### 文档解析

请参考 [PaddleOCR-VL 使用教程](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.md) 中的客户端调用相关章节。

### 健康检查

```bash
# 存活检查
curl http://localhost:8080/health

# 就绪检查（验证 Triton 和 VLM 服务是否已准备好处理请求）
curl http://localhost:8080/health/ready
```

## 性能调优

### 并发设置

网关对推理操作和非推理操作各自独立地进行并发控制：

- **`HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`**（默认 16）：控制 `layout-parsing`（版面解析）等推理操作的并发数
  - 过低（4）：推理设备利用率不足，请求不必要地排队
  - 过高（64）：可能导致 Triton 过载，出现 OOM 或超时
  - 默认值 16 允许在当前批次处理时有足够请求排队形成下一批次
  - 如推理设备资源有限，建议适当降低此值
- **`HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS`**（默认 64）：控制 `restructure-pages`（多页重组）等非推理操作的并发数
  - 非推理操作不占用推理设备资源，可以设置更高的并发数
  - 可根据 CPU 核数和内存情况调整

**高吞吐配置示例：**

```bash
# .env
HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=32
HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS=128
UVICORN_WORKERS=8
```

**低延迟配置示例：**

```bash
# .env
HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=8
HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS=32
HPS_INFERENCE_TIMEOUT=300
UVICORN_WORKERS=2
```

### Worker 进程数

每个 Uvicorn Worker 是独立的进程，有自己的事件循环：

- **1 个 Worker**：简单，但受限于单进程
- **4 个 Worker**：适合大多数场景
- **8+ 个 Worker**：适用于高并发、大量小请求的场景

### Triton 动态批处理

Triton 自动将请求批处理以提高推理设备利用率。最大批处理大小通过模型配置文件中的 `max_batch_size` 参数控制（默认：8），配置文件位于模型仓库目录下的 `config.pbtxt`（如 `model_repo/layout-parsing/config.pbtxt`）。

## 故障排查与解决

### 服务无法启动

查看各服务的日志以定位问题：

```bash
docker compose logs paddleocr-vl-api
docker compose logs paddleocr-vl-tritonserver
docker compose logs paddleocr-vlm-server
```

常见原因包括端口被占用、推理设备不可用或镜像拉取失败。

### 超时错误

- 增加 `HPS_INFERENCE_TIMEOUT`（针对复杂文档）
- 如果推理设备过载，减少 `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`

### 内存/显存不足

- 减少 `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`
- 确保每个推理设备只运行一个服务
- 检查 compose.yaml 中的 `shm_size`（默认：4GB）
