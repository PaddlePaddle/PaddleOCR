---
comments: true
---

# Installation

## 1. Install PaddleOCR Inference Dependencies

### 1.1 Install the Inference Engine

PaddleOCR 3.5 introduces a unified inference-engine configuration concept. At the backend, it supports different inference engines such as the PaddlePaddle framework and Transformers. If you want to run pipeline/model inference, refer to [Inference Engine and Configuration](./inference_engine.en.md) to install the required inference engine. If you only want to use features other than PaddleOCR inference, you can skip this step.

### 1.2 Install the PaddleOCR Inference Package

Install the latest PaddleOCR inference package from PyPI:

```bash
# If you only want to use the basic text recognition feature
# (returns text coordinates and recognized text)
python -m pip install paddleocr
# If you want to use all features, such as document parsing,
# document understanding, document translation, and key information extraction
# python -m pip install "paddleocr[all]"
```

Or install from source (the development branch is used by default):

```bash
# If you only want to use the basic text recognition feature
# (returns text coordinates and recognized text)
python -m pip install "paddleocr@git+https://github.com/PaddlePaddle/PaddleOCR.git"
# If you want to use all features, such as document parsing,
# document understanding, document translation, and key information extraction
# python -m pip install "paddleocr[all]@git+https://github.com/PaddlePaddle/PaddleOCR.git"
```

In addition to the `all` dependency group shown above, PaddleOCR also supports installing selected optional capabilities through other dependency groups. PaddleOCR provides the following dependency groups:

| Dependency Group Name | Corresponding Functionality |
| - | - |
| `doc-parser` | Document parsing. Used to extract layout elements such as tables, formulas, seals, and images from documents. Includes model solutions such as PP-StructureV3 |
| `ie` | Information extraction. Used to extract key information such as names, dates, addresses, and amounts from documents. Includes model solutions such as PP-ChatOCRv4 |
| `trans` | Document translation. Used to translate documents from one language to another. Includes model solutions such as PP-DocTranslation |
| `all` | Full functionality |

The general OCR pipeline (such as PP-OCRv3/v4/v5) and the document image preprocessing pipeline can be used without installing any additional dependency groups. Besides these two pipelines, each pipeline belongs to exactly one dependency group. You can check the usage documentation of each pipeline to see which dependency group it belongs to. For individual modules, installing any dependency group that contains the corresponding module is sufficient for basic usage.

## 2. Install Training Dependencies

If you want to perform development tasks such as model training and export, you need to install the training dependencies. It is allowed to install both the inference package and the training dependencies in the same environment. No environment isolation is required.

Training and export workflows depend on the PaddlePaddle framework. First complete the PaddlePaddle installation by following [PaddlePaddle Framework Installation](./paddlepaddle_installation.en.md).

To perform model training, exporting, and similar tasks, clone the repository to your local machine:

```bash
# Recommended method
git clone https://github.com/PaddlePaddle/PaddleOCR

# (Optional) Switch to a specific branch
git checkout release/3.5

# If cloning fails because of network issues, you can also use the Gitee repository:
git clone https://gitee.com/paddlepaddle/PaddleOCR

# Note: The code hosted on Gitee may lag behind the GitHub repository by 3 to 5 days.
# Please prioritize the recommended method.
```

Run the following command to install the remaining training dependencies:

```bash
python -m pip install -r requirements.txt
```
