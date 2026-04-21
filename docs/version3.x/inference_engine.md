---
comments: true
---

# 推理引擎与配置说明

PaddleOCR 3.5 引入了统一的推理引擎配置方式：使用 `engine` 选择底层推理引擎，使用 `engine_config` 传递该引擎的专属配置。无论是单模型还是产线，均可按这套方式声明推理行为。

如果不显式指定 `engine`，默认行为与旧版本保持一致：除高性能推理功能和生成式 AI 客户端请求功能等少数场景外，大多数情况下会优先使用飞桨框架推理；如果显式指定 `engine`，则会优先按指定引擎初始化。

## 1. 什么是推理引擎

在 PaddleOCR 中，推理引擎指模型执行时所使用的底层运行时。它决定了模型由哪套运行时加载与执行，可以将它理解为“模型推理时实际使用的引擎”。在使用推理引擎时，用户通常只需要关心两件事：

- 选择哪类推理引擎；
- 如何配置推理引擎。

## 2. PaddleOCR 当前支持的推理引擎

| 引擎类别 | `engine` 取值 | 说明 |
| - | - | - |
| 飞桨框架 | `paddle`、`paddle_static`、`paddle_dynamic` | 基于飞桨框架运行。 |
| Transformers | `transformers` | 基于 Hugging Face Transformers 运行。 |

- `paddle`：飞桨框架统一入口。根据模型类型和模型目录中的文件选择 `paddle_static` 或 `paddle_dynamic`，在二者都可用的情况下偏好 `paddle_static`。
- `paddle_static`：飞桨静态图推理，适合对推理性能有一定要求或者需要进行精细化推理性能调优的场景。
- `paddle_dynamic`：飞桨动态图推理，相比静态图更加灵活、易于调试。
- `transformers`：Hugging Face Transformers 推理，便于与 Hugging Face 生态集成。

## 3. 各推理引擎安装方式

### 3.1 飞桨框架

当您使用飞桨框架进行推理时，需要先安装飞桨框架。安装方法请参考[飞桨框架安装](./paddlepaddle_installation.md)。

### 3.2 Transformers

当您使用 Transformers 作为推理引擎时，需要安装 Hugging Face Transformers。示例命令如下：

```bash
python -m pip install "transformers>=5.4.0"
```

通常，您还需要安装底层推理框架，详情可参考 [Transformers 官方文档](https://huggingface.co/docs/transformers/installation)。

## 4. `engine` 和 `engine_config` 的设置与取值

### 4.1 `engine`

`engine` 用于指定推理引擎，可取值如下：

| 取值 | 含义 | 说明 |
| - | - | - |
| `None` | 不显式指定引擎 | 自动确定推理引擎。保持 PaddleOCR 3.4 的行为，大多数情况下会使用飞桨框架推理。 |
| `paddle` | 飞桨框架统一入口 | 自动选择 `paddle_static` 或 `paddle_dynamic`。 |
| `paddle_static` | 静态图推理 | 使用飞桨静态图推理。 |
| `paddle_dynamic` | 飞桨动态图推理 | 使用飞桨动态图推理。 |
| `transformers` | Transformers 推理 | 使用 Hugging Face Transformers 推理。 |

### 4.2 `engine_config`

`engine_config` 用于配置推理引擎，建议与 `engine` 搭配使用。各引擎常见的 `engine_config` 字段如下：

#### `paddle_static`

常见字段包括：

- `run_mode`：运行模式，如 `paddle`、`trt_fp32`、`trt_fp16`、`mkldnn`；
- `device_type` / `device_id`：设备类型和设备编号；
- `cpu_threads`：CPU 推理线程数；
- `delete_pass`：手动禁用的图优化 pass 列表；
- `enable_new_ir`：是否启用新 IR；
- `enable_cinn`：是否启用 CINN；
- `trt_cfg_setting`：TensorRT 底层配置；
- `trt_use_dynamic_shapes`：是否启用 TensorRT 动态形状；
- `trt_collect_shape_range_info`：是否采集 shape range 信息；
- `trt_discard_cached_shape_range_info`：是否丢弃已有 shape range 信息并重新采集；
- `trt_dynamic_shapes`：动态形状配置；
- `trt_dynamic_shape_input_data`：采集动态形状时用于填充输入张量的数据；
- `trt_shape_range_info_path`：shape range 信息文件路径；
- `trt_allow_rebuild_at_runtime`：运行时是否允许重建 TensorRT 引擎；
- `mkldnn_cache_capacity`：oneDNN（MKLDNN）缓存容量。

#### `paddle_dynamic`

常见字段包括：

- `device_type` / `device_id`：动态图执行时的设备类型和设备编号。

#### `transformers`

常见字段包括：

- `dtype`：模型权重 / 推理使用的数据类型，如 `float16`；
- `device_type` / `device_id`：推理设备类型和设备编号；
- `trust_remote_code`：是否信任并执行模型仓库中的自定义代码；
- `attn_implementation`：注意力实现方式，如 `flash_attention_2`；
- `generation_config`：生成参数，如 `max_new_tokens`、`temperature`；
- `model_kwargs`：传给模型加载接口的额外参数；
- `processor_kwargs`：传给 processor / image processor 加载接口的额外参数；
- `tokenizer_kwargs`：兼容保留字段，会与 `processor_kwargs` 合并使用。

#### 4.2.1 扁平与分桶 `engine_config`

同一层级的 `engine_config` 可以是：

- **扁平**：只包含**当前解析得到的引擎**所需的字段（例如仅使用静态图时，顶层直接是 `run_mode`、`cpu_threads` 等）。
- **分桶**：顶层键**仅**为 PaddleX 已注册的引擎名（如 `paddle_static`、`paddle_dynamic`、`transformers` 等），每个键对应一个嵌套字典。**不得**在同一层级混用「分桶键」与扁平字段（例如 `{"paddle_static": {...}, "run_mode": "paddle"}` 会报错）。

解析为某一引擎时，只会使用与该引擎对应的一份配置：扁平形式直接参与校验；分桶形式则取出对应键下的字典。

### 4.3 优先级与覆盖规则

- 对于产线，命令行参数或 Python API 初始化参数中传入的 `engine`、`engine_config`，优先级高于产线配置文件中的同名字段；
- 在产线配置文件中，顶层 `engine`、`engine_config` 会作为全局配置，子模块或子产线中的 `engine`、`engine_config` 可覆盖上层配置；
- 关于产线配置文件中更完整的优先级、覆盖和调用规则，建议参考 PaddleX 文档：[PaddleX 产线 Python 脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/instructions/pipeline_python_API.html)。

### 4.4 兼容性规则

- 显式设置 `engine` 后，`enable_hpi` 不再生效；
- 显式传入 `engine_config` 后，与该引擎对应的兼容参数会被忽略。例如在 `paddle` / `paddle_static` 场景下，`use_tensorrt`、`precision`、`enable_mkldnn`、`mkldnn_cache_capacity`、`cpu_threads`、`enable_cinn` 等兼容参数不再生效。

## 5. 调用示例

### 5.1 单模型（CLI）：通过 `--engine` 选择引擎

```bash
paddleocr text_detection -i general_ocr_001.png --engine transformers
```

### 5.2 单模型（Python）：显式指定 `transformers`

```python
from paddleocr import TextDetection

model = TextDetection(
    model_name="PP-OCRv5_server_det",
    engine="transformers",
)

result = model.predict("general_ocr_001.png")
```

### 5.3 单模型（Python）：指定 `paddle_static` 与 `engine_config`

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

### 5.4 产线（CLI）：通过 `--engine` 选择引擎

```bash
paddleocr ocr -i general_ocr_001.png --engine paddle_static
```

### 5.5 产线（Python API）：为某个模块单独配置推理引擎

如需为产线中的某一个模块单独指定 `engine`、`engine_config`，可先导出配置文件，修改对应模块配置后，再通过加载配置文件。配置文件的导出、编辑与加载方式可参见 [使用 PaddleX 产线配置文件](./paddleocr_and_paddlex.md#3-paddlex)。示例如下：

首先，导出产线配置文件：

```python
from paddleocr import PaddleOCR

pipeline = PaddleOCR()
pipeline.export_paddlex_config_to_yaml("ocr_config.yaml")
```

然后，在 `ocr_config.yaml` 中为 `TextDetection` 模块单独设置 `engine`、`engine_config`：

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

使用更新后的配置文件完成推理：

```python
from paddleocr import PaddleOCR

pipeline = PaddleOCR(paddlex_config="ocr_config.yaml")
result = pipeline.predict("general_ocr_001.png")
```
