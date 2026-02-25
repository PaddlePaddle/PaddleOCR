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


## 📖 文档

完整文档请参阅 [API 参考](https://reference.langchain.com/python/integrations/langchain_paddleocr/)。有关概念指南、教程和使用示例，请参阅 [LangChain 文档](https://docs.langchain.com/oss/python/integrations/providers/paddleocr)。
