# langchain-paddleocr

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-paddleocr?label=%20)](https://pypi.org/project/langchain-paddleocr/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-paddleocr)](https://opensource.org/license/apache-2-0)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-paddleocr)](https://pypistats.org/packages/langchain-paddleocr)

This package provides access to PaddleOCR's capabilities within the LangChain ecosystem.

## Quick Install

```bash
pip install langchain-paddleocr
```

## Basic Usage

### `PaddleOCRVLLoader`

The `PaddleOCRVLLoader` enables you to:

- Extract text and layout information from PDF and image files using models from Baidu's PaddleOCR-VL series (e.g., PaddleOCR-VL, PaddleOCR-VL-1.5)
- Process documents from local files or remote URLs

Basic usage of `PaddleOCRVLLoader` looks as follows:

```python
from langchain_paddleocr import PaddleOCRVLLoader
from pydantic import SecretStr

loader = PaddleOCRVLLoader(
    file_path="path/to/document.pdf",
    api_url="your-api-endpoint",
    access_token=SecretStr("your-access-token")  # Optional if using environment variable `PADDLEOCR_ACCESS_TOKEN`
)

docs = loader.load()

for doc in docs[:2]:
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Source: {doc.metadata['source']}")
    print("---")
```

### `PaddleOCRLoader`

The `PaddleOCRLoader` wraps the **local** PaddleOCR library to extract text from PDF and image files — no cloud API or access token required.

It supports two modes:

- **Basic OCR** (default) — fast text extraction using PP-OCRv5.
- **Structure mode** — layout-aware extraction (tables, titles, figures) using PP-StructureV3.

#### Basic OCR

```python
from langchain_paddleocr import PaddleOCRLoader

loader = PaddleOCRLoader(file_path="path/to/document.pdf")
docs = loader.load()

for doc in docs:
    print(f"Page {doc.metadata['page']}: {doc.page_content[:100]}...")
    print(f"Confidence: {doc.metadata['confidence']:.2f}")
```

#### Structure mode

```python
from langchain_paddleocr import PaddleOCRLoader
from langchain_paddleocr.document_loaders.paddleocr import PaddleOCRConfig

config = PaddleOCRConfig(lang="en", use_table_recognition=True)
loader = PaddleOCRLoader(
    file_path=["page1.png", "page2.png"],
    use_structure=True,
    config=config,
)

for doc in loader.lazy_load():
    print(doc.page_content)
    print(doc.metadata["layout_blocks"])
```

#### Configuration

Use `PaddleOCRConfig` to pass engine parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lang` | `str` | Language code (`"ch"`, `"en"`, `"fr"`, etc.) |
| `ocr_version` | `str` | Pipeline version (`"PP-OCRv3"`, `"PP-OCRv4"`, `"PP-OCRv5"`) |
| `use_doc_orientation_classify` | `bool` | Enable document orientation classification |
| `use_doc_unwarping` | `bool` | Enable document de-warping |
| `text_det_thresh` | `float` | Detection confidence threshold |
| `text_rec_score_thresh` | `float` | Recognition confidence threshold |
| `use_table_recognition` | `bool` | Enable table recognition (structure mode) |
| `use_chart_recognition` | `bool` | Enable chart recognition (structure mode) |

See the full list in `PaddleOCRConfig`.

## 📖 Documentation

For full documentation, see the [LangChain Docs](https://docs.langchain.com/oss/python/integrations/providers/baidu).
