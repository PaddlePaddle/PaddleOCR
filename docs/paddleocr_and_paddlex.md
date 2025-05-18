# PaddleOCR 与 PaddleX

[PaddleX](https://github.com/PaddlePaddle/PaddleX) 是一款基于飞桨框架构建的低代码开发工具，集成了众多开箱即用的预训练模型，支持模型从训练到推理的全流程开发，兼容多款国内外主流硬件，助力 AI 开发者在产业实践中高效落地。

PaddleOCR 在推理部署方面基于 PaddleX 构建，二者在该环节可实现无缝协同。在安装 PaddleOCR 时，PaddleX 也将作为其依赖一并安装。此外，PaddleOCR 与 PaddleX 在产线名称等方面也保持一致。对于快速体验，如果只使用基础配置，用户通常无需了解 PaddleX 的具体概念；但在涉及高级配置、服务化部署等使用场景时，了解 PaddleX 的相关知识将有所帮助。

本文档将介绍 PaddleOCR 与 PaddleX 之间的关系，并说明如何协同使用这两个工具。

## 1. PaddleOCR 与 PaddleX 的区别与联系

PaddleOCR 与 PaddleX 在定位和功能上各有侧重：PaddleOCR 专注于 OCR 相关任务，而 PaddleX 则覆盖了包括时序预测、人脸识别等在内的多种任务类型。此外，PaddleX 提供了丰富的基础设施，具备多模型组合推理的底层能力，能够以统一且灵活的方式接入不同模型，支持构建复杂的模型产线。

PaddleOCR 在推理部署环节充分复用了 PaddleX 的能力，具体包括：

- PaddleOCR 在模型推理、前后处理及多模型组合等底层能力上，主要依赖于 PaddleX。
- PaddleOCR 的高性能推理能力通过 PaddleX 的 Paddle2ONNX 插件及高性能推理插件实现。
- PaddleOCR 的服务化部署方案基于 PaddleX 的实现。

## 2. PaddleOCR 产线与 PaddleX 产线注册名的对应关系

| PaddleOCR 产线 | PaddleX 产线注册名 |
| --- | --- |
| 通用 OCR | `OCR` |
| 通用版面解析 v3 | `PP-StructureV3` |
| 文档场景信息抽取 v4 | `PP-ChatOCRv4-doc` |
| 通用表格识别 v2 | `table_recognition_v2` |
| 公式识别 | `formula_recognition` |
| 印章文本识别 | `seal_recognition` |
| 文档图像预处理 | `doc_preprocessor` |
| 文档理解 | `doc_understanding` |

## 3. 使用 PaddleX 产线配置文件

在推理部署阶段，PaddleOCR 支持导出和加载 PaddleX 的产线配置文件。用户可通过编辑配置文件，对推理部署相关参数进行深度配置。

### 3.1 导出产线配置文件

可调用 PaddleOCR 产线对象的 `export_paddlex_config_to_yaml` 方法，将当前产线配置导出为 YAML 文件。示例如下：

```python
from paddleocr import PaddleOCR

pipeline = PaddleOCR()
pipeline.export_paddlex_config_to_yaml("ocr_config.yaml")
```

上述代码会在工作目录下生成名为 `ocr_config.yaml` 的产线配置文件。

### 3.2 编辑产线配置文件

导出的 PaddleX 产线配置文件不仅包含 PaddleOCR CLI 和 Python API 支持的参数，还可进行更多高级配置。请在 [PaddleX模型产线使用概览](https://paddlepaddle.github.io/PaddleX/3.0/pipeline_usage/pipeline_develop_guide.html) 中找到对应的产线使用教程，参考其中的详细说明，根据需求调整各项配置。

### 3.3 在 CLI 中加载产线配置文件

通过 `--paddlex_config` 参数指定 PaddleX 产线配置文件的路径，PaddleOCR 会读取其中的内容作为产线的默认配置。示例如下：

```bash
paddleocr ocr --paddlex_config ocr_config.yaml ...
```

### 3.4 在 Python API 中加载产线配置文件

初始化产线对象时，可通过 `paddlex_config` 参数传入 PaddleX 产线配置文件路径或配置字典，PaddleOCR 会将其作为默认配置。示例如下：

```python
from paddleocr import PaddleOCR

pipeline = PaddleOCR(paddlex_config="ocr_config.yaml")
```
