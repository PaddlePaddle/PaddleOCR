---
comments: true
---

# Inference Engine and Configuration

PaddleOCR 3.5 introduces a unified inference-engine configuration mechanism: use `engine` to select the underlying inference engine, and use `engine_config` to pass engine-specific settings. This mechanism applies to both individual models and pipelines.

If `engine` is not explicitly specified, the default behavior remains the same as in earlier versions: except for a few scenarios such as high-performance inference and generative AI client request features, PaddleOCR uses the PaddlePaddle framework for inference in most cases. If `engine` is explicitly specified, initialization follows the selected engine first.

## 1. What Is an Inference Engine

In PaddleOCR, an inference engine refers to the underlying runtime used to execute a model. It determines which runtime loads and runs the model. You can think of it as "the engine actually used during model inference." When using an inference engine, users usually only need to care about two things:

- which type of inference engine to use;
- how to configure the inference engine.

## 2. Inference Engines Currently Supported by PaddleOCR

| Engine category | `engine` values | Description |
| - | - | - |
| PaddlePaddle framework | `paddle`, `paddle_static`, `paddle_dynamic` | Runs on the PaddlePaddle framework. |
| Transformers | `transformers` | Runs on Hugging Face Transformers. |

- `paddle`: The unified entry point for the PaddlePaddle framework. It selects `paddle_static` or `paddle_dynamic` according to the model type and files in the model directory. If both are available, `paddle_static` is preferred.
- `paddle_static`: PaddlePaddle static-graph inference, suitable for scenarios that require better inference performance or more fine-grained performance tuning.
- `paddle_dynamic`: PaddlePaddle dynamic-graph inference, which is more flexible and easier to debug compared to static graph.
- `transformers`: Hugging Face Transformers inference, making it convenient to integrate with the Hugging Face ecosystem.

## 3. Installation by Inference Engine

### 3.1 PaddlePaddle framework

When using the PaddlePaddle framework for inference, you need to install PaddlePaddle first. For installation instructions, see [PaddlePaddle Framework Installation](./paddlepaddle_installation.en.md).

### 3.2 Transformers

When using Transformers as the inference engine, you need to install Hugging Face Transformers. Example command:

```bash
python -m pip install "transformers>=5.4.0"
```

In many cases, you also need to install the underlying inference framework. For details, see the [Transformers official documentation](https://huggingface.co/docs/transformers/installation).

## 4. Configuration and Supported Values of `engine` and `engine_config`

### 4.1 `engine`

`engine` is used to specify the inference engine. Supported values are:

| Value | Meaning | Description |
| - | - | - |
| `None` | No explicit engine specified | Automatically determines the inference engine. Keep the behavior of PaddleOCR 3.4; in most cases, the PaddlePaddle framework will be used for inference. |
| `paddle` | Unified PaddlePaddle framework entry | Automatically selects `paddle_static` or `paddle_dynamic`. |
| `paddle_static` | Static-graph inference | Uses Paddle static-graph inference. |
| `paddle_dynamic` | Dynamic-graph inference | Uses Paddle dynamic-graph inference. |
| `transformers` | Transformers inference | Uses Hugging Face Transformers inference. |

### 4.2 `engine_config`

`engine_config` is used to configure the inference engine and is recommended to be used together with `engine`. Common `engine_config` fields for each engine are listed below:

#### `paddle_static`

Common fields include:

- `run_mode`: execution mode, such as `paddle`, `trt_fp32`, `trt_fp16`, and `mkldnn`;
- `device_type` / `device_id`: device type and device index;
- `cpu_threads`: number of CPU inference threads;
- `delete_pass`: list of graph optimization passes to disable manually;
- `enable_new_ir`: whether to enable the new IR;
- `enable_cinn`: whether to enable CINN;
- `trt_cfg_setting`: low-level TensorRT configuration;
- `trt_use_dynamic_shapes`: whether to enable TensorRT dynamic shapes;
- `trt_collect_shape_range_info`: whether to collect shape range information;
- `trt_discard_cached_shape_range_info`: whether to discard existing shape range information and recollect it;
- `trt_dynamic_shapes`: dynamic shape configuration;
- `trt_dynamic_shape_input_data`: data used to fill input tensors when collecting dynamic shapes;
- `trt_shape_range_info_path`: path to the shape range information file;
- `trt_allow_rebuild_at_runtime`: whether rebuilding the TensorRT engine is allowed at runtime;
- `mkldnn_cache_capacity`: oneDNN (MKLDNN) cache capacity.

#### `paddle_dynamic`

Common fields include:

- `device_type` / `device_id`: device type and device index used during dynamic-graph execution.

#### `transformers`

Common fields include:

- `dtype`: data type used for model weights / inference, such as `float16`;
- `device_type` / `device_id`: inference device type and device index;
- `trust_remote_code`: whether to trust and execute custom code in model repositories;
- `attn_implementation`: attention implementation method, such as `flash_attention_2`;
- `generation_config`: generation parameters, such as `max_new_tokens` and `temperature`;
- `model_kwargs`: extra arguments passed to the model loading API;
- `processor_kwargs`: extra arguments passed to the processor / image processor loading API;
- `tokenizer_kwargs`: a compatibility-preserved field that is merged with `processor_kwargs`.

#### 4.2.1 Flat vs. bucketed `engine_config`

At the same level `engine_config` may be:

- **Flat**: a dict whose keys are only those required by the **resolved** engine (for example, when using static graph only, top-level keys such as `run_mode` and `cpu_threads`).
- **Bucketed**: top-level keys are **only** registered engine names (e.g. `paddle_static`, `paddle_dynamic`, `transformers`), each mapping to a nested dict. You **must not** mix bucket keys with flat keys at the same level (e.g. `{"paddle_static": {...}, "run_mode": "paddle"}` is invalid).

When an engine is resolved, only the corresponding config is used: flat configs are validated as a whole; bucketed configs take the entry for that engine.

### 4.3 Priority and Override Rules

- For pipelines, `engine` and `engine_config` passed through CLI arguments or Python API initialization arguments take precedence over fields with the same names in the pipeline configuration file.
- In pipeline configuration files, top-level `engine` and `engine_config` act as global settings; `engine` and `engine_config` in submodules or sub-pipelines can override upper-level settings.
- For more complete rules about priority, overriding, and pipeline configuration behavior, refer to the PaddleX documentation: [PaddleX Pipeline Python API Usage](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/instructions/pipeline_python_API.html).

### 4.4 Compatibility Rules

- When `engine` is explicitly set, `enable_hpi` no longer takes effect.
- When `engine_config` is explicitly provided, compatibility arguments for the selected engine are ignored. For example, in `paddle` / `paddle_static` scenarios, compatibility arguments such as `use_tensorrt`, `precision`, `enable_mkldnn`, `mkldnn_cache_capacity`, `cpu_threads`, and `enable_cinn` no longer take effect.

## 5. Usage Examples

### 5.1 Individual model (CLI): select the engine with `--engine`

```bash
paddleocr text_detection -i general_ocr_001.png --engine transformers
```

### 5.2 Individual model (Python): explicitly specify `transformers`

```python
from paddleocr import TextDetection

model = TextDetection(
    model_name="PP-OCRv5_server_det",
    engine="transformers",
)

result = model.predict("general_ocr_001.png")
```

### 5.3 Individual model (Python): specify `paddle_static` and `engine_config`

```python
from paddleocr import TextDetection

model = TextDetection(
    model_name="PP-OCRv5_server_det",
    engine="paddle_static",
    engine_config={
        "device_type": "cpu",
        "cpu_threads": 4,
        "run_mode": "mkldnn",
    },
)

result = model.predict("general_ocr_001.png")
```

### 5.4 Pipeline (CLI): select the engine with `--engine`

```bash
paddleocr ocr -i general_ocr_001.png --engine paddle_static
```

### 5.5 Pipeline (Python API): configure the inference engine for a specific module

If you want to specify `engine` and `engine_config` for a specific module inside a pipeline, you can first export the configuration file, modify the corresponding module configuration, and then load it. For how to export, edit, and load the configuration file, see [Using PaddleX Pipeline Configuration Files](./paddleocr_and_paddlex.en.md#3-using-paddlex-pipeline-configuration-files). Example:

First, export the pipeline configuration file:

```python
from paddleocr import PaddleOCR

pipeline = PaddleOCR()
pipeline.export_paddlex_config_to_yaml("ocr_config.yaml")
```

Then, set `engine` and `engine_config` specifically for the `TextDetection` module in `ocr_config.yaml`:

```yaml
pipeline_name: OCR
SubModules:
  TextDetection:
    engine: paddle_static
    engine_config:
      device_type: cpu
      cpu_threads: 4
      run_mode: mkldnn
```

Use the updated configuration file for inference:

```python
from paddleocr import PaddleOCR

pipeline = PaddleOCR(paddlex_config="ocr_config.yaml")
result = pipeline.predict("general_ocr_001.png")
```
