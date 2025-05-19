# PaddleOCR and PaddleX

[PaddleX](https://github.com/PaddlePaddle/PaddleX) is a low-code development tool built on the PaddlePaddle framework. It integrates numerous out-of-the-box pre-trained models, supports the full-pipeline development from model training to inference, and is compatible with various mainstream hardware both domestically and internationally, empowering AI developers to efficiently deploy solutions in industrial practices.

PaddleOCR leverages PaddleX for inference deployment, enabling seamless collaboration between the two in this regard. When installing PaddleOCR, PaddleX is also installed as a dependency. Additionally, PaddleOCR and PaddleX maintain consistency in pipeline naming conventions. For quick experience, users typically do not need to understand the specific concepts of PaddleX when using basic configurations. However, knowledge of PaddleX can be beneficial in advanced configuration scenarios, service deployment, and other use cases.

This document introduces the relationship between PaddleOCR and PaddleX and explains how to use these two tools collaboratively.

## 1. Differences and Connections Between PaddleOCR and PaddleX

PaddleOCR and PaddleX have distinct focuses and functionalities: PaddleOCR specializes in OCR-related tasks, while PaddleX covers a wide range of task types, including time-series forecasting, face recognition, and more. Furthermore, PaddleX provides rich infrastructure with underlying capabilities for multi-model combined inference, enabling the integration of different models in a unified and flexible manner and supporting the construction of complex model pipelines.

PaddleOCR fully reuses the capabilities of PaddleX in the inference deployment phase, including:

- PaddleOCR primarily relies on PaddleX for underlying capabilities such as model inference, pre- and post-processing, and multi-model combination.
- The high-performance inference capabilities of PaddleOCR are achieved through PaddleX's Paddle2ONNX plugin and high-performance inference plugins.
- The service deployment solutions of PaddleOCR are based on PaddleX's implementations.

## 2. Correspondence Between PaddleOCR Pipelines and PaddleX Pipeline Registration Names

| PaddleOCR Pipeline | PaddleX Pipeline Registration Name |
| --- | --- |
| General OCR | `OCR` |
| General Layout Analysis v3 | `PP-StructureV3` |
| Document Scenario Information Extraction v4 | `PP-ChatOCRv4-doc` |
| General Table Recognition v2 | `table_recognition_v2` |
| Formula Recognition | `formula_recognition` |
| Seal Text Recognition | `seal_recognition` |
| Document Image Preprocessing | `doc_preprocessor` |
| Document Understanding | `doc_understanding` |

## 3. Using PaddleX Pipeline Configuration Files

During the inference deployment phase, PaddleOCR supports exporting and loading PaddleX pipeline configuration files. Users can deeply configure inference deployment-related parameters by editing these configuration files.

### 3.1 Exporting Pipeline Configuration Files

You can call the `export_paddlex_config_to_yaml` method of the PaddleOCR pipeline object to export the current pipeline configuration to a YAML file. Here is an example:

```python
from paddleocr import PaddleOCR

pipeline = PaddleOCR()
pipeline.export_paddlex_config_to_yaml("ocr_config.yaml")
```

The above code will generate a pipeline configuration file named `ocr_config.yaml` in the working directory.

### 3.2 Editing Pipeline Configuration Files

The exported PaddleX pipeline configuration file not only includes parameters supported by PaddleOCR's CLI and Python API but also allows for more advanced configurations. Please refer to the corresponding pipeline usage tutorials in [PaddleX Pipeline Usage Overview](https://paddlepaddle.github.io/PaddleX/3.0/en/pipeline_usage/pipeline_develop_guide.html) for detailed instructions on adjusting various configurations according to your needs.

### 3.3 Loading Pipeline Configuration Files in CLI

By specifying the path to the PaddleX pipeline configuration file using the `--paddlex_config` parameter, PaddleOCR will read its contents as the default configuration for the pipeline. Here is an example:

```bash
paddleocr ocr --paddlex_config ocr_config.yaml ...
```

### 3.4 Loading Pipeline Configuration Files in Python API

When initializing the pipeline object, you can pass the path to the PaddleX pipeline configuration file or a configuration dictionary through the `paddlex_config` parameter, and PaddleOCR will use it as the default configuration. Here is an example:

```python
from paddleocr import PaddleOCR

pipeline = PaddleOCR(paddlex_config="ocr_config.yaml")
```
