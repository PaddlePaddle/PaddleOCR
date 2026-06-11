# langchain-paddleocr

[![PyPI - 版本](https://img.shields.io/pypi/v/langchain-paddleocr?label=%20)](https://pypi.org/project/langchain-paddleocr/#history)
[![PyPI - 许可证](https://img.shields.io/pypi/l/langchain-paddleocr)](https://opensource.org/license/apache-2-0)
[![PyPI - 下载量](https://img.shields.io/pepy/dt/langchain-paddleocr)](https://pypistats.org/packages/langchain-paddleocr)

本 Python 包在 LangChain 生态系统中提供对 PaddleOCR 功能的访问。

## 快速安装

```bash
pip install langchain-paddleocr
```

## 基本用法

### `PaddleOCRVLLoader`

`PaddleOCRVLLoader` 允许你：

- 使用百度 PaddleOCR-VL 系列模型（例如 PaddleOCR-VL、PaddleOCR-VL-1.5）从 PDF 和图像文件中提取文本和版面布局信息
- 处理来自本地文件或远程 URL 的文档

`PaddleOCRVLLoader` 的基本用法如下：

```python
from langchain_paddleocr import PaddleOCRVLLoader
from pydantic import SecretStr

loader = PaddleOCRVLLoader(
    file_path="path/to/document.pdf",
    api_url="your-api-endpoint",
    access_token=SecretStr("your-access-token")  # 如果使用环境变量 `PADDLEOCR_ACCESS_TOKEN`，则此项为可选
)

docs = loader.load()

for doc in docs[:2]:
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Source: {doc.metadata['source']}")
    print("---")
```


### `PaddleOCRLoader`

`PaddleOCRLoader` 封装了 **本地** PaddleOCR 库，从 PDF 和图像文件中提取文本 — 无需云 API 或访问令牌。

支持两种模式：

- **基础 OCR**（默认）— 使用 PP-OCRv5 进行快速文本提取。
- **版面分析模式** — 使用 PP-StructureV3 进行版面感知提取（表格、标题、图片等）。

#### 基础 OCR

```python
from langchain_paddleocr import PaddleOCRLoader

loader = PaddleOCRLoader(file_path="path/to/document.pdf")
docs = loader.load()

for doc in docs:
    print(f"页面 {doc.metadata['page']}: {doc.page_content[:100]}...")
    print(f"置信度: {doc.metadata['confidence']:.2f}")
```

#### 版面分析模式

```python
from langchain_paddleocr import PaddleOCRLoader
from langchain_paddleocr.document_loaders.paddleocr import PaddleOCRConfig

config = PaddleOCRConfig(lang="ch", use_table_recognition=True)
loader = PaddleOCRLoader(
    file_path=["page1.png", "page2.png"],
    use_structure=True,
    config=config,
)

for doc in loader.lazy_load():
    print(doc.page_content)
    print(doc.metadata["layout_blocks"])
```

#### 配置

使用 `PaddleOCRConfig` 传递引擎参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `lang` | `str` | 语言代码（`"ch"`、`"en"`、`"fr"` 等） |
| `ocr_version` | `str` | 流水线版本（`"PP-OCRv3"`、`"PP-OCRv4"`、`"PP-OCRv5"`） |
| `use_doc_orientation_classify` | `bool` | 启用文档方向分类 |
| `use_doc_unwarping` | `bool` | 启用文档去弯曲 |
| `text_det_thresh` | `float` | 检测置信度阈值 |
| `text_rec_score_thresh` | `float` | 识别置信度阈值 |
| `use_table_recognition` | `bool` | 启用表格识别（版面分析模式） |
| `use_chart_recognition` | `bool` | 启用图表识别（版面分析模式） |

完整参数请参阅 `PaddleOCRConfig`。

## 📖 文档

完整文档请参阅 [LangChain 文档](https://docs.langchain.com/oss/python/integrations/providers/baidu)。
