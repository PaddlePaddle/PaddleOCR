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

## 📖 Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/integrations/langchain_paddleocr/). For conceptual guides, tutorials, and usage examples, see the [LangChain Docs](https://docs.langchain.com/oss/python/integrations/providers/paddleocr).
