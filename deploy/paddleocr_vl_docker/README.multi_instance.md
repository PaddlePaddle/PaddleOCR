# 多实例部署与负载均衡示例

本目录下提供了 `compose.multi_instance.yaml.example` 与 `nginx.multi_instance.conf`，用于演示如何在 **单个 vLLM 推理服务** 之上，通过多个 `paddlex --serve` API 实例来提升并发处理能力。请根据自身 GPU / 服务器资源调整配置。

## 1. 准备工作

1. **配置产线参数**
   - 为每个 API 实例准备的产线配置中，将 `VLRecognition.genai_config.server_url` 指向同一个 vLLM 服务。
   - 调整 `VLRecognition.genai_config.max_concurrency`，确保单个 vLLM 显存可以容纳该并发（通常 4~16 之间，视显存与 `max-num-seqs` 而定）。
   - vLLM 端的 `backend_config`（如 `max-num-seqs`、`gpu-memory-utilization`）也需要同步修改，允许来自多个 API 的请求同时执行。

2. **Nginx（可选）**
   - `nginx.multi_instance.conf` 通过轮询方式将请求均衡到多个 API 实例。
   - 该配置已增加超时时间与请求体大小限制，适合大 PDF / 图像推理场景，可根据实际情况继续调整。

## 2. 启动示例

1. 复制示例 compose 文件，并根据需求修改端口、设备 ID 等：

   ```bash
   cp deploy/paddleocr_vl_docker/compose.multi_instance.yaml.example compose.yaml
   ```

2. （可选）如需使用 Nginx 负载均衡，将配置复制到当前目录：

   ```bash
   cp deploy/paddleocr_vl_docker/nginx.multi_instance.conf nginx.conf
   ```

3. 启动服务：

   ```bash
   docker compose up -d
   ```

   - `paddleocr-vlm-server`：单个 vLLM 推理服务，供多个 API 共用。
   - `paddleocr-vl-api-*`：多个 `paddlex --serve` 实例，分别暴露不同端口（示例中为 8081、8082 等）。
   - `paddleocr-vl-nginx-lb`（可选）：监听 8080，将请求轮询到后台 API。

4. 客户端调用
   - 直接访问某个 API 端口（例如 `http://host:8081`），或通过 Nginx 暴露的统一入口（例如 `http://host:8080`）。

## 3. 扩展建议

- **增加 API 实例**：复制 `paddleocr-vl-api-*` 服务定义并修改端口即可。记得同步更新 Nginx upstream 列表。
- **GPU 资源隔离**：若单卡无法满足多 API 并发，可按 GPU 拆分多个 vLLM 服务，每张卡构建一对 “API + VLM”，再使用上层负载均衡。
- **监控与稳定性**：建议结合容器 healthcheck、Prometheus/Grafana 等监控方案，持续观察显存/延迟，必要时自动重启异常实例。

按照上述步骤即可快速启动“多 API + 单 VLM”的并行推理集群，适用于批量 PDF/图片的高并发处理场景。根据生产环境需求，继续扩容或调整配置即可。
