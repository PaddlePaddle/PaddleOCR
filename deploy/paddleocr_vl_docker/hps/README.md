# PaddleOCR-VL 高性能服务化部署（Beta）

本目录提供一套支持并发请求处理的 PaddleOCR-VL 高性能服务化部署方案。

## 环境要求

- x64 CPU
- NVIDIA GPU，Compute Capability >= 8.0 且 < 12.0
- NVIDIA 驱动支持 CUDA 12.6
- Docker >= 19.03

## 快速开始

拉取 PaddleOCR 源码并切换到当前目录：

```shell
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd deploy/paddleocr_vl_docker/hps
```

下载并拷贝必要文件到当前目录：

```shell
bash prepare.sh
```

启动服务：

```shell
docker compose up
```

上述命令将依次启动 3 个容器，每个容器对应一个服务：

- **`paddleocr-vlm-server`**：基于 vLLM 的 VLM 推理服务。
- **`paddleocr-vl-tritonserver`**：基于 Triton Inference Server 的 PaddleOCR-VL 产线推理服务。
- **`paddleocr-vl-api`**：使用 FastAPI 实现的网关服务，用于将 HTTP 请求转发至 Triton Inference Server，并封装返回结果，简化客户端调用流程。**该服务为对外入口**，客户端可直接通过 HTTP 调用。

> 首次启动会自动下载并构建镜像，耗时较长；从第二次启动起将直接使用本地镜像，启动速度更快。

## 调整服务配置

tbd
