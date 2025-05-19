# Serving

Serving is a common deployment method in real-world production environments. By encapsulating inference capabilities as services, clients can access these services via network requests to obtain inference results. PaddleOCR recommends using [PaddleX](https://github.com/PaddlePaddle/PaddleX) for serving. Please refer to [Differences and Connections between PaddleOCR and PaddleX](../paddleocr_and_paddlex.en.md#1-Differences-and-Connections-Between-PaddleOCR-and-PaddleX) to understand the relationship between PaddleOCR and PaddleX.

PaddleX provides the following serving solutions:

- **Basic Serving**: An easy-to-use serving solution with low development costs.
- **High-Stability Serving**: Built based on [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server). Compared to the basic serving, this solution offers higher stability and allows users to adjust configurations to optimize performance.

**It is recommended to first use the basic serving solution for quick validation**, and then evaluate whether to try more complex solutions based on actual needs.

## 1. Basic Serving

### 1.1 Install Dependencies

Run the following command to install the PaddleX serving plugin via PaddleX CLI:

```bash
paddlex --install serving
```

### 1.2 Run the Server

Run the server via PaddleX CLI:

```bash
paddlex --serve --pipeline {PaddleX pipeline registration name or pipeline configuration file path} [{other command-line options}]
```

Take the general OCR pipeline as an example:

```bash
paddlex --serve --pipeline OCR
```

You should see information similar to the following:

```text
INFO:     Started server process [63108]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

To adjust configurations (such as model path, batch size, deployment device, etc.), specify `--pipeline` as a custom configuration file. Refer to [PaddleOCR and PaddleX](../paddleocr_and_paddlex.en.md) for the mapping between PaddleOCR pipelines and PaddleX pipeline registration names, as well as how to obtain and modify PaddleX pipeline configuration files.

The command-line options related to serving are as follows:

<table>
<thead>
<tr>
<th>Name</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>--pipeline</code></td>
<td>PaddleX pipeline registration name or pipeline configuration file path.</td>
</tr>
<tr>
<td><code>--device</code></td>
<td>Deployment device for the pipeline. Defaults to <code>cpu</code> (if GPU is unavailable) or <code>gpu</code> (if GPU is available).</td>
</tr>
<tr>
<td><code>--host</code></td>
<td>Hostname or IP address to which the server is bound. Defaults to <code>0.0.0.0</code>.</td>
</tr>
<tr>
<td><code>--port</code></td>
<td>Port number on which the server listens. Defaults to <code>8080</code>.</td>
</tr>
<tr>
<td><code>--use_hpip</code></td>
<td>If specified, uses high-performance inference.</td>
</tr>
<tr>
<td><code>--hpi_config</code></td>
<td>High-performance inference configuration. Refer to the <a href="https://paddlepaddle.github.io/PaddleX/3.0/en/pipeline_deploy/high_performance_inference.html#22">PaddleX High-Performance Inference Guide</a> for more information.</td>
</tr>
</tbody>
</table>

### 1.3 Invoke the Service

The <b>"Development Integration/Deployment"</b> section in the PaddleOCR pipeline tutorial provides API references and multi-language invocation examples for the service.

## 2. High-Stability Serving

Please refer to the [PaddleX Serving Guide](https://paddlepaddle.github.io/PaddleX/3.0/en/pipeline_deploy/serving.html#2). More information about PaddleX pipeline configuration files can be found in [Using PaddleX Pipeline Configuration Files](../paddleocr_and_paddlex.en.md#3-using-paddlex-pipeline-configuration-files).

It should be noted that, due to the lack of fine-grained optimization and other reasons, the current high-stability serving deployment solution provided by PaddleOCR may not match the performance of the 2.x version based on PaddleServing. However, this new solution fully supports the PaddlePaddle 3.0 framework. We will continue to optimize it and consider introducing more performant deployment solutions in the future.
