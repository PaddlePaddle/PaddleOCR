# 服务化部署

服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleOCR 推荐用户使用 [PaddleX](https://github.com/PaddlePaddle/PaddleX) 进行服务化部署。请阅读 [PaddleOCR 与 PaddleX 的区别与联系](../paddleocr_and_paddlex.md#1-paddleocr-与-paddlex-的区别与联系) 了解 PaddleOCR 与 PaddleX 的关系。

PaddleX 提供以下服务化部署方案：

- **基础服务化部署**：简单易用的服务化部署方案，开发成本低。
- **高稳定性服务化部署**：基于 [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) 打造。与基础服务化部署相比，该方案提供更高的稳定性，并允许用户调整配置以优化性能。

**建议首先使用基础服务化部署方案进行快速验证**，然后根据实际需要，评估是否尝试更复杂的方案。

## 1. 基础服务化部署

### 1.1 安装依赖

执行如下命令，通过 PaddleX CLI 安装 PaddleX 服务化部署插件：

```bash
paddlex --install serving
```

### 1.2 运行服务器

通过 PaddleX CLI 运行服务器：

```bash
paddlex --serve --pipeline {PaddleX 产线注册名或产线配置文件路径} [{其他命令行选项}]
```

以通用 OCR 产线为例：

```bash
paddlex --serve --pipeline OCR
```

可以看到类似以下展示的信息：

```text
INFO:     Started server process [63108]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

如需调整配置（如模型路径、batch size、部署设备等），可指定 `--pipeline` 为自定义配置文件。请参考 [PaddleOCR 与 PaddleX](../paddleocr_and_paddlex.md) 了解 PaddleOCR 产线与 PaddleX 产线注册名的对应关系，以及 PaddleX 产线配置文件的获取与修改方式。

与服务化部署相关的命令行选项如下：

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
<td>产线部署设备。默认为 <code>cpu</code>（如 GPU 不可用）或 <code>gpu</code>（如 GPU 可用）。</td>
</tr>
<tr>
<td><code>--host</code></td>
<td>服务器绑定的主机名或 IP 地址。默认为 <code>0.0.0.0</code>。</td>
</tr>
<tr>
<td><code>--port</code></td>
<td>服务器监听的端口号。默认为 <code>8080</code>。</td>
</tr>
<tr>
<td><code>--use_hpip</code></td>
<td>如果指定，则使用高性能推理。</td>
</tr>
<tr>
<td><code>--hpi_config</code></td>
<td>高性能推理配置。请参考 <a href="https://paddlepaddle.github.io/PaddleX/3.0/pipeline_deploy/high_performance_inference.html#22">PaddleX 高性能推理指南</a> 了解更多信息。</td>
</tr>
</tbody>
</table>
</table>

### 1.3 调用服务

PaddleOCR 产线使用教程中的 <b>“开发集成/部署”</b> 部分提供了服务的 API 参考与多语言调用示例。

## 2. 高稳定性服务化部署

请参考 [PaddleX 服务化部署指南](https://paddlepaddle.github.io/PaddleX/3.0/pipeline_deploy/serving.html#2)。在 [使用 PaddleX 产线配置文件](../paddleocr_and_paddlex.md#3-使用-paddlex-产线配置文件) 中，可以了解关于 PaddleX 产线配置文件的更多信息。

需要说明的是，由于缺乏细粒度优化等原因，当前 PaddleOCR 提供的高稳定性服务化部署方案在性能上可能不及 2.x 版本基于 PaddleServing 的方案；但该新方案已对飞桨 3.0 框架提供了全面支持，我们也将持续优化，后续考虑推出性能更优的部署方案。
