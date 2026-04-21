---
comments: true
---

# Installation

## 1. Install the inference engine, PaddleOCR Python package, and optional feature dependencies

This section explains how to install the inference engine as needed, the `paddleocr` distribution package, and optional dependency groups by capability domain. This path covers running pretrained pipelines for inference locally, as well as auxiliary features such as document format conversion. **Model training and model export** are covered in Section 2 and are independent of the installation path above.

### 1.1 Install the inference engine (as needed)

PaddleOCR 3.5 uses a unified inference-engine configuration and can use backends such as PaddlePaddle and Transformers. To actually run model inference, install your chosen inference engine by following [Inference Engine and Configuration](./inference_engine.en.md).

### 1.2 Install paddleocr

Install the latest `paddleocr` from PyPI:

```bash
# Default capabilities only: general OCR and document image preprocessing
python -m pip install paddleocr
# All optional capabilities: document parsing, document understanding,
# document translation, key information extraction, etc.
# python -m pip install "paddleocr[all]"
```

Or install from source (tracks the repository’s current default branch by default):

```bash
# Default capabilities only: general OCR and document image preprocessing
python -m pip install "paddleocr@git+https://github.com/PaddlePaddle/PaddleOCR.git"
# All optional capabilities: document parsing, document understanding,
# document translation, key information extraction, etc.
# python -m pip install "paddleocr[all]@git+https://github.com/PaddlePaddle/PaddleOCR.git"
```

### 1.3 Choose dependency groups by capability

Besides `all`, you can enable selected optional capabilities by specifying dependency groups. Each group corresponds to a capability domain (document parsing, information extraction, document translation, etc.). The available groups are:

| Dependency Group Name | Corresponding Functionality |
| - | - |
| `doc-parser` | Document parsing. Extracts layout elements such as tables, formulas, seals, and images from documents. Includes model solutions such as PP-StructureV3 |
| `ie` | Information extraction. Extracts key information such as names, dates, addresses, and amounts from documents. Includes model solutions such as PP-ChatOCRv4 |
| `trans` | Document translation. Translates documents from one language to another. Includes model solutions such as PP-DocTranslation |
| `doc2md` | Document-to-Markdown conversion. Quickly turns Word, Excel, and PowerPoint files into readable text |
| `all` | Full functionality |

The general OCR pipeline and the document image preprocessing pipeline require no extra dependency groups; document parsing, information extraction, document translation, and other capabilities follow the table above. See each pipeline’s documentation for its dependency group. For individual modules, install any dependency group that contains the module to use its basic functionality.

## 2. Install training and export dependencies

To train models or export models, install the training-related dependencies separately. This path is a different installation dimension from the `paddleocr` package and optional groups in Section 1; both can coexist in one environment without mandatory isolation.

Training and export depend on the PaddlePaddle framework. Complete PaddlePaddle installation first by following [PaddlePaddle Framework Installation](./paddlepaddle_installation.en.md).

Clone this repository locally, then install the remaining dependencies:

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
