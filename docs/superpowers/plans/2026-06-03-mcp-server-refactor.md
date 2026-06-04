# MCP Server 架构重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 PaddleOCR MCP Server 以使用官方 SDK 的 AsyncPaddleOCRClient，实现 Executor + MCPCapability 两层架构。

**Architecture:** 将当前单一的 PipelineHandler 拆分为两个抽象层：Executor 负责底层推理执行，MCPCapability 负责 MCP 工具注册和结果格式化。

**Tech Stack:** Python 3.10+, FastMCP, httpx, paddleocr SDK (AsyncPaddleOCRClient)

---

## 文件结构

**新建文件：**
- `mcp_server/paddleocr_mcp/executors/__init__.py` - Executor 工厂函数
- `mcp_server/paddleocr_mcp/executors/base.py` - Executor 抽象基类
- `mcp_server/paddleocr_mcp/executors/http.py` - HTTPExecutor 基类
- `mcp_server/paddleocr_mcp/executors/local.py` - LocalExecutor
- `mcp_server/paddleocr_mcp/executors/aistudio.py` - AIStudioExecutor（主要新功能）
- `mcp_server/paddleocr_mcp/executors/qianfan.py` - QianfanExecutor
- `mcp_server/paddleocr_mcp/executors/self_hosted.py` - SelfHostedExecutor
- `mcp_server/paddleocr_mcp/capabilities/__init__.py` - Capability 工厂函数
- `mcp_server/paddleocr_mcp/capabilities/base.py` - MCPCapability 抽象基类
- `mcp_server/paddleocr_mcp/capabilities/ocr.py` - OCRCapability
- `mcp_server/paddleocr_mcp/capabilities/layout.py` - LayoutParsingCapability

**修改文件：**
- `mcp_server/paddleocr_mcp/__init__.py` - 更新导出
- `mcp_server/paddleocr_mcp/__main__.py` - 使用新架构
- `mcp_server/pyproject.toml` - 更新版本号到 0.7.0

**删除文件：**
- `mcp_server/paddleocr_mcp/pipelines.py` - 被新架构替代

---

## Task 1: 创建 executors 模块基础结构

**Files:**
- Create: `mcp_server/paddleocr_mcp/executors/__init__.py`
- Create: `mcp_server/paddleocr_mcp/executors/base.py`

- [ ] **Step 1: 创建 executors/__init__.py**

```python
# mcp_server/paddleocr_mcp/executors/__init__.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base import Executor, ExecutorError, AuthenticationError, ResourceUnavailableError, TimeoutError
from .http import HTTPExecutor
from .local import LocalExecutor
from .aistudio import AIStudioExecutor
from .qianfan import QianfanExecutor
from .self_hosted import SelfHostedExecutor
from .factory import create_executor

__all__ = [
    "Executor",
    "ExecutorError",
    "AuthenticationError",
    "ResourceUnavailableError",
    "TimeoutError",
    "HTTPExecutor",
    "LocalExecutor",
    "AIStudioExecutor",
    "QianfanExecutor",
    "SelfHostedExecutor",
    "create_executor",
]
```

- [ ] **Step 2: 创建 executors/base.py（Executor 抽象基类）**

```python
# mcp_server/paddleocr_mcp/executors/base.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Any, Dict, Optional


class ExecutorError(RuntimeError):
    """Executor 执行错误基类"""
    pass


class AuthenticationError(ExecutorError):
    """认证失败"""
    pass


class ResourceUnavailableError(ExecutorError):
    """服务不可用"""
    pass


class TimeoutError(ExecutorError):
    """超时"""
    pass


class Executor(abc.ABC):
    """执行器抽象基类，负责底层推理执行"""

    @abc.abstractmethod
    async def execute(
        self,
        input_data: str,
        file_type: Optional[str] = None,
        **options
    ) -> Dict[str, Any]:
        """执行推理，返回统一格式的结果

        Args:
            input_data: 输入数据（文件路径、URL 或 base64）
            file_type: 文件类型（"image" 或 "pdf"）
            **options: 其他选项

        Returns:
            统一格式的结果字典
        """
        pass

    @abc.abstractmethod
    async def start(self) -> None:
        """初始化资源"""
        pass

    @abc.abstractmethod
    async def stop(self) -> None:
        """清理资源"""
        pass
```

- [ ] **Step 3: 提交**

```bash
cd /Users/linmanhui/Repos/PaddleOCR
git add mcp_server/paddleocr_mcp/executors/__init__.py mcp_server/paddleocr_mcp/executors/base.py
git commit -m "refactor(mcp): add Executor base class and module structure"
```

---

## Task 2: 创建 HTTPExecutor 基类

**Files:**
- Create: `mcp_server/paddleocr_mcp/executors/http.py`

- [ ] **Step 1: 创建 executors/http.py**

```python
# mcp_server/paddleocr_mcp/executors/http.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import asyncio
import json
from typing import Any, Dict, Optional

import httpx

from .base import Executor, AuthenticationError, ExecutorError, ResourceUnavailableError


class HTTPExecutor(Executor):
    """同步 HTTP API 的抽象基类，封装通用的 HTTP 调用逻辑"""

    def __init__(self, base_url: str, timeout: int = 60):
        self._base_url = base_url
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        timeout = httpx.Timeout(connect=30.0, read=self._timeout, write=30.0, pool=30.0)
        self._client = httpx.AsyncClient(timeout=timeout)

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _post(self, endpoint: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """执行 HTTP POST 请求"""
        url = f"{self._base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        try:
            response = await self._client.post(url, json=payload, headers=headers)
            if response.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {response.text}")
            if response.status_code == 503:
                raise ResourceUnavailableError(f"Service unavailable: {response.text}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise ExecutorError(f"HTTP request failed: {e}")
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            raise ExecutorError(f"HTTP request failed: {e}")

    @abc.abstractmethod
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头（子类实现认证）"""
        pass

    @abc.abstractmethod
    def _get_endpoint(self) -> str:
        """获取 API 端点"""
        pass

    @abc.abstractmethod
    def _prepare_payload(self, input_data: str, file_type: Optional[str], **options) -> Dict[str, Any]:
        """准备请求 payload（子类实现）"""
        pass

    @abc.abstractmethod
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """解析响应为统一格式（子类实现）"""
        pass

    async def execute(self, input_data: str, file_type: Optional[str] = None, **options) -> Dict[str, Any]:
        headers = self._get_headers()
        payload = self._prepare_payload(input_data, file_type, **options)
        response = await self._post(self._get_endpoint(), payload, headers)
        return self._parse_response(response)
```

- [ ] **Step 2: 提交**

```bash
cd /Users/linmanhui/Repos/PaddleOCR
git add mcp_server/paddleocr_mcp/executors/http.py
git commit -m "refactor(mcp): add HTTPExecutor base class for synchronous HTTP APIs"
```

---

## Task 3: 创建 AIStudioExecutor（核心新功能）

**Files:**
- Create: `mcp_server/paddleocr_mcp/executors/aistudio.py`

- [ ] **Step 1: 创建 executors/aistudio.py**

```python
# mcp_server/paddleocr_mcp/executors/aistudio.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
from typing import Any, Dict, Optional

from paddleocr._api_client.async_client import AsyncPaddleOCRClient
from paddleocr._api_client.errors import (
    APIError,
    AuthError,
    JobFailedError,
    RequestTimeoutError,
    ResponseFormatError,
    ResultParseError,
    ServiceUnavailableError,
)
from paddleocr._api_client.models import Model, OCROptions, DocParsingOptions

from .base import AuthenticationError, Executor, ExecutorError, ResourceUnavailableError, TimeoutError


class AIStudioExecutor(Executor):
    """使用 PaddleOCR SDK 的 AI Studio 执行器"""

    # Pipeline 到 SDK Model 的映射
    _PIPELINE_MODEL_MAP = {
        "OCR": Model.PP_OCRV5,
        "PP-StructureV3": Model.PP_STRUCTURE_V3,
        "PaddleOCR-VL": Model.PADDLE_OCR_VL,
        "PaddleOCR-VL-1.5": Model.PADDLE_OCR_VL_15,
        "PaddleOCR-VL-1.6": Model.PADDLE_OCR_VL_16,
    }

    # Pipeline 类型映射
    _OCR_PIPELINES = {"OCR"}
    _DOC_PARSING_PIPELINES = {"PP-StructureV3", "PaddleOCR-VL", "PaddleOCR-VL-1.5", "PaddleOCR-VL-1.6"}

    def __init__(
        self,
        pipeline: str,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: float = 300.0,
        poll_timeout: float = 600.0,
    ):
        self._pipeline = pipeline
        self._token = token
        self._base_url = base_url
        self._request_timeout = request_timeout
        self._poll_timeout = poll_timeout
        self._client: Optional[AsyncPaddleOCRClient] = None

        if pipeline not in self._PIPELINE_MODEL_MAP:
            raise ValueError(f"Unknown pipeline: {pipeline}")

    async def start(self) -> None:
        try:
            self._client = AsyncPaddleOCRClient(
                token=self._token,
                base_url=self._base_url,
                request_timeout=self._request_timeout,
                poll_timeout=self._poll_timeout,
            )
        except AuthError as e:
            raise AuthenticationError(f"Authentication failed: {e}")

    async def stop(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    def _resolve_model(self) -> Model:
        """获取 pipeline 对应的 Model"""
        return self._PIPELINE_MODEL_MAP[self._pipeline]

    def _resolve_input_source(self, input_data: str):
        """解析输入源（URL 或文件路径）"""
        if input_data.startswith("http://") or input_data.startswith("https://"):
            return {"file_url": input_data}
        else:
            return {"file_path": input_data}

    async def execute(self, input_data: str, file_type: Optional[str] = None, **options) -> Dict[str, Any]:
        if not self._client:
            raise RuntimeError("Executor not started")

        model = self._resolve_model()
        input_source = self._resolve_input_source(input_data)

        try:
            if self._pipeline in self._OCR_PIPELINES:
                # OCR 调用
                ocr_options = OCROptions(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    visualize=options.get("visualize", False),
                )
                result = await self._client.ocr(
                    model=model,
                    **input_source,
                    options=ocr_options,
                )
                return self._parse_ocr_result(result)

            elif self._pipeline in self._DOC_PARSING_PIPELINES:
                # 文档解析调用
                doc_options = DocParsingOptions(
                    use_layout_detection=options.get("use_layout_detection", True),
                    use_chart_recognition=options.get("use_chart_recognition", True),
                    temperature=options.get("temperature"),
                    prettify_markdown=options.get("prettify_markdown", True),
                )
                result = await self._client.parse_document(
                    model=model,
                    **input_source,
                    options=doc_options,
                )
                return self._parse_doc_parsing_result(result)

        except AuthError as e:
            raise AuthenticationError(f"Authentication failed: {e}")
        except ServiceUnavailableError as e:
            raise ResourceUnavailableError(f"Service unavailable: {e}")
        except (JobFailedError, APIError, ResponseFormatError, ResultParseError) as e:
            raise ExecutorError(f"Execution failed: {e}")
        except RequestTimeoutError as e:
            raise TimeoutError(f"Request timeout: {e}")

    def _parse_ocr_result(self, result) -> Dict[str, Any]:
        """将 SDK OCRResult 解析为统一格式"""
        clean_texts, confidences, text_lines = [], [], []

        for page_result in result.pages:
            for line in page_result.text_lines:
                text = line.text
                conf = line.confidence
                bbox = line.bounding_box  # SDK 返回的 bbox 格式

                if text and text.strip():
                    clean_texts.append(text.strip())
                    confidences.append(conf)
                    text_lines.append({
                        "text": text.strip(),
                        "confidence": round(conf, 3),
                        "bbox": bbox,
                    })

        return {
            "text": "\n".join(clean_texts),
            "confidence": sum(confidences) / len(confidences) if confidences else 0,
            "text_lines": text_lines,
        }

    def _parse_doc_parsing_result(self, result) -> Dict[str, Any]:
        """将 SDK DocParsingResult 解析为统一格式"""
        markdown_parts = []
        all_images_mapping = {}

        for page in result.pages:
            markdown_parts.append(page.markdown)
            # 处理图片
            for img_key, img_url in page.images.items():
                all_images_mapping[img_key] = img_url

        return {
            "markdown": "\n".join(markdown_parts),
            "pages": len(result.pages),
            "images_mapping": all_images_mapping,
        }
```

- [ ] **Step 2: 提交**

```bash
cd /Users/linmanhui/Repos/PaddleOCR
git add mcp_server/paddleocr_mcp/executors/aistudio.py
git commit -m "feat(mcp): add AIStudioExecutor using AsyncPaddleOCRClient"
```

---

## Task 4: 创建 LocalExecutor

**Files:**
- Create: `mcp_server/paddleocr_mcp/executors/local.py`

- [ ] **Step 1: 创建 executors/local.py**

```python
# mcp_server/paddleocr_mcp/executors/local.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import io
from pathlib import PurePath
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import puremagic
from PIL import Image as PILImage

from .base import Executor, ExecutorError

try:
    from paddleocr import PaddleOCR, PaddleOCRVL, PPStructureV3
    LOCAL_OCR_AVAILABLE = True
except ImportError:
    LOCAL_OCR_AVAILABLE = False


class _EngineWrapper:
    """包装本地推理引擎，使其可以在异步上下文中运行"""
    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._queue: Queue = Queue()
        self._closed = False
        self._loop = asyncio.get_running_loop()
        self._thread = Thread(target=self._worker, daemon=False)
        self._thread.start()

    @property
    def engine(self) -> Any:
        return self._engine

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("Engine wrapper has already been closed")
        fut = self._loop.create_future()
        self._queue.put((func, args, kwargs, fut))
        return await fut

    async def close(self) -> None:
        if not self._closed:
            self._queue.put(None)
            await self._loop.run_in_executor(None, self._thread.join)
            self._closed = True

    def _worker(self) -> None:
        while not self._closed:
            item = self._queue.get()
            if item is None:
                break
            func, args, kwargs, fut = item
            try:
                result = func(*args, **kwargs)
                self._loop.call_soon_threadsafe(fut.set_result, result))
            except Exception as e:
                self._loop.call_soon_threadsafe(fut.set_exception(e))
            finally:
                self._queue.task_done()


class LocalExecutor(Executor):
    """本地 PaddleOCR 执行器"""

    def __init__(
        self,
        pipeline: str,
        pipeline_config: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self._pipeline = pipeline
        self._pipeline_config = pipeline_config
        self._device = device
        self._engine: Optional[Any] = None
        self._engine_wrapper: Optional[_EngineWrapper] = None

    async def start(self) -> None:
        if not LOCAL_OCR_AVAILABLE:
            raise RuntimeError("PaddleOCR is not locally available")
        try:
            self._engine = self._create_engine()
            self._engine_wrapper = _EngineWrapper(self._engine)
        except Exception as e:
            raise RuntimeError(f"Failed to create PaddleOCR engine: {str(e)}") from e

    async def stop(self) -> None:
        if self._engine_wrapper:
            await self._engine_wrapper.close()
            self._engine_wrapper = None

    def _create_engine(self) -> Any:
        """根据 pipeline 类型创建对应的引擎"""
        if self._pipeline == "OCR":
            return PaddleOCR(
                paddlex_config=self._pipeline_config,
                device=self._device,
            )
        elif self._pipeline == "PP-StructureV3":
            return PPStructureV3(
                paddlex_config=self._pipeline_config,
                device=self._device,
            )
        elif self._pipeline == "PaddleOCR-VL":
            return PaddleOCRVL(
                pipeline_version="v1",
                paddlex_config=self._pipeline_config,
                device=self._device,
            )
        elif self._pipeline == "PaddleOCR-VL-1.5":
            return PaddleOCRVL(
                pipeline_version="v1.5",
                paddlex_config=self._pipeline_config,
                device=self._device,
            )
        elif self._pipeline == "PaddleOCR-VL-1.6":
            return PaddleOCRVL(
                pipeline_version="v1.6",
                paddlex_config=self._pipeline_config,
                device=self._device,
            )
        else:
            raise ValueError(f"Unknown pipeline: {self._pipeline}")

    def _is_file_path(self, s: str) -> bool:
        try:
            PurePath(s)
            return True
        except Exception:
            return False

    def _is_url(self, s: str) -> bool:
        if not (s.startswith("http://") or s.startswith("https://")):
            return False
        from urllib.parse import urlparse
        result = urlparse(s)
        return all([result.scheme, result.netloc]) and result.scheme in ("http", "https")

    def _is_base64(self, s: str) -> bool:
        import re
        pattern = r"^[A-Za-z0-9+/]+={0,2}$"
        return bool(re.fullmatch(pattern, s))

    def _infer_file_type_from_bytes(self, data: bytes) -> Optional[str]:
        mime = puremagic.from_string(data, mime=True)
        if mime.startswith("image/"):
            return "image"
        elif mime == "application/pdf":
            return "pdf"
        return None

    def _process_input_for_local(self, input_data: str) -> Union[str, np.ndarray]:
        """为本地处理准备输入"""
        if self._is_base64(input_data):
            if input_data.startswith("data:"):
                base64_data = input_data.split(",", 1)[1]
            else:
                base64_data = input_data
            try:
                image_bytes = base64.b64decode(base64_data)
                file_type = self._infer_file_type_from_bytes(image_bytes)
                if file_type != "image":
                    raise ValueError("Currently, only images can be passed via Base64.")
                image_pil = PILImage.open(io.BytesIO(image_bytes))
                image_arr = np.array(image_pil.convert("RGB"))
                return np.ascontiguousarray(image_arr[..., ::-1])
            except Exception as e:
                raise ValueError(f"Failed to decode Base64 image: {str(e)}") from e
        elif self._is_file_path(input_data) or self._is_url(input_data):
            return input_data
        else:
            raise ValueError("Invalid input data format")

    async def execute(self, input_data: str, file_type: Optional[str] = None, **options) -> Dict[str, Any]:
        if not self._engine_wrapper:
            raise RuntimeError("Engine wrapper not initialized")

        processed_input = self._process_input_for_local(input_data)

        # 调用推理
        result = await self._engine_wrapper.call(
            self._engine_wrapper.engine.predict, processed_input
        )

        return self._parse_result(result)

    def _parse_result(self, result: Any) -> Dict[str, Any]:
        """解析本地推理结果为统一格式"""
        if self._pipeline == "OCR":
            return self._parse_ocr_result(result)
        else:
            return self._parse_layout_result(result)

    def _parse_ocr_result(self, result: Any) -> Dict[str, Any]:
        """解析 OCR 结果"""
        clean_texts, confidences, text_lines = [], [], []

        for res in result:
            texts = res["rec_texts"]
            scores = res["rec_scores"]
            boxes = res["rec_boxes"]

            for i, text in enumerate(texts):
                if text and text.strip():
                    conf = scores[i] if i < len(scores) else 0
                    clean_texts.append(text.strip())
                    confidences.append(conf)
                    text_lines.append({
                        "text": text.strip(),
                        "confidence": round(conf, 3),
                        "bbox": boxes[i].tolist(),
                    })

        return {
            "text": "\n".join(clean_texts),
            "confidence": sum(confidences) / len(confidences) if confidences else 0,
            "text_lines": text_lines,
        }

    def _parse_layout_result(self, result: Any) -> Dict[str, Any]:
        """解析版面解析结果"""
        markdown_parts = []
        all_images_mapping = {}

        for res in result:
            markdown = res.markdown
            text = markdown["markdown_texts"]
            markdown_parts.append(text)
            images = markdown["markdown_images"]
            processed_images = {}
            for img_key, img_data in images.items():
                with io.BytesIO() as buffer:
                    img_data.save(buffer, format="JPEG")
                    processed_images[img_key] = base64.b64encode(buffer.getvalue()).decode("ascii")
            all_images_mapping.update(processed_images)

        return {
            "markdown": "\n".join(markdown_parts),
            "pages": len(result),
            "images_mapping": all_images_mapping,
        }
```

- [ ] **Step 2: 提交**

```bash
cd /Users/linmanhui/Repos/PaddleOCR
git add mcp_server/paddleocr_mcp/executors/local.py
git commit -m "refactor(mcp): add LocalExecutor for local inference"
```

---

## Task 5: 创建 QianfanExecutor 和 SelfHostedExecutor

**Files:**
- Create: `mcp_server/paddleocr_mcp/executors/qianfan.py`
- Create: `mcp_server/paddleocr_mcp/executors/self_hosted.py`

- [ ] **Step 1: 创建 executors/qianfan.py**

```python
# mcp_server/paddleocr_mcp/executors/qianfan.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional

from .http import HTTPExecutor


class QianfanExecutor(HTTPExecutor):
    """千帆平台执行器"""

    def __init__(self, base_url: str, api_key: str, pipeline: str, timeout: int = 60):
        super().__init__(base_url, timeout)
        self._api_key = api_key
        self._pipeline = pipeline

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _get_endpoint(self) -> str:
        # 千帆统一使用 paddleocr 端点
        return "paddleocr"

    def _prepare_payload(self, input_data: str, file_type: Optional[str], **options) -> Dict[str, Any]:
        payload = {"file": input_data}
        if file_type == "image":
            payload["fileType"] = 1
        elif file_type == "pdf":
            payload["fileType"] = 0
        return payload

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        # 千帆响应格式解析（与自托管类似）
        result_data = response.get("result", response)

        if self._pipeline == "OCR":
            return self._parse_ocr_response(result_data)
        else:
            return self._parse_layout_response(result_data)

    def _parse_ocr_response(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        ocr_results = result_data.get("ocrResults", [])
        all_texts, all_confidences, text_lines = [], [], []

        for ocr_result in ocr_results:
            pruned = ocr_result["prunedResult"]
            texts = pruned["rec_texts"]
            scores = pruned["rec_scores"]
            boxes = pruned["rec_boxes"]

            for i, text in enumerate(texts):
                if text and text.strip():
                    conf = scores[i] if i < len(scores) else 0
                    all_texts.append(text.strip())
                    all_confidences.append(conf)
                    text_lines.append({
                        "text": text.strip(),
                        "confidence": round(conf, 3),
                        "bbox": boxes[i],
                    })

        return {
            "text": "\n".join(all_texts),
            "confidence": sum(all_confidences) / len(all_confidences) if all_confidences else 0,
            "text_lines": text_lines,
        }

    def _parse_layout_response(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        layout_results = result_data.get("layoutParsingResults", [])
        markdown_parts = []
        all_images_mapping = {}

        for res in layout_results:
            markdown_parts.append(res["markdown"]["text"])
            images = res["markdown"]["images"]
            all_images_mapping.update(images)

        return {
            "markdown": "\n".join(markdown_parts),
            "pages": len(layout_results),
            "images_mapping": all_images_mapping,
        }
```

- [ ] **Step 2: 创建 executors/self_hosted.py**

```python
# mcp_server/paddleocr_mcp/executors/self_hosted.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional

from .http import HTTPExecutor


class SelfHostedExecutor(HTTPExecutor):
    """自托管服务执行器"""

    def __init__(self, base_url: str, pipeline: str, timeout: int = 60):
        super().__init__(base_url, timeout)
        self._pipeline = pipeline

    def _get_headers(self) -> Dict[str, str]:
        return {}  # 无需认证

    def _get_endpoint(self) -> str:
        if self._pipeline == "OCR":
            return "ocr"
        else:
            return "layout-parsing"

    def _prepare_payload(self, input_data: str, file_type: Optional[str], **options) -> Dict[str, Any]:
        payload = {"file": input_data}
        if file_type == "image":
            payload["fileType"] = 1
        elif file_type == "pdf":
            payload["fileType"] = 0
        return payload

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        result_data = response.get("result", response)

        if self._pipeline == "OCR":
            return self._parse_ocr_response(result_data)
        else:
            return self._parse_layout_response(result_data)

    def _parse_ocr_response(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        ocr_results = result_data.get("ocrResults", [])
        all_texts, all_confidences, text_lines = [], [], []

        for ocr_result in ocr_results:
            pruned = ocr_result["prunedResult"]
            texts = pruned["rec_texts"]
            scores = pruned["rec_scores"]
            boxes = pruned["rec_boxes"]

            for i, text in enumerate(texts):
                if text and text.strip():
                    conf = scores[i] if i < len(scores) else 0
                    all_texts.append(text.strip())
                    all_confidences.append(conf)
                    text_lines.append({
                        "text": text.strip(),
                        "confidence": round(conf, 3),
                        "bbox": boxes[i],
                    })

        return {
            "text": "\n".join(all_texts),
            "confidence": sum(all_confidences) / len(all_confidences) if all_confidences else 0,
            "text_lines": text_lines,
        }

    def _parse_layout_response(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        layout_results = result_data.get("layoutParsingResults", [])
        markdown_parts = []
        all_images_mapping = {}

        for res in layout_results:
            markdown_parts.append(res["markdown"]["text"])
            images = res["markdown"]["images"]
            all_images_mapping.update(images)

        return {
            "markdown": "\n".join(markdown_parts),
            "pages": len(layout_results),
            "images_mapping": all_images_mapping,
        }
```

- [ ] **Step 3: 提交**

```bash
cd /Users/linmanhui/Repos/PaddleOCR
git add mcp_server/paddleocr_mcp/executors/qianfan.py mcp_server/paddleocr_mcp/executors/self_hosted.py
git commit -m "refactor(mcp): add QianfanExecutor and SelfHostedExecutor"
```

---

## Task 6: 创建 Executor 工厂函数

**Files:**
- Create: `mcp_server/paddleocr_mcp/executors/factory.py`
- Modify: `mcp_server/paddleocr_mcp/executors/__init__.py`

- [ ] **Step 1: 创建 executors/factory.py**

```python
# mcp_server/paddleocr_mcp/executors/factory.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from .aistudio import AIStudioExecutor
from .base import Executor
from .local import LocalExecutor
from .qianfan import QianfanExecutor
from .self_hosted import SelfHostedExecutor


def create_executor(
    source: str,
    pipeline: str,
    token: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 60,
    api_key: Optional[str] = None,
    pipeline_config: Optional[str] = None,
    device: Optional[str] = None,
) -> Executor:
    """根据 source 创建对应的 Executor

    Args:
        source: 执行器来源（local, aistudio, qianfan, self_hosted）
        pipeline: Pipeline 类型
        token: AI Studio 访问令牌（aistudio 模式）
        base_url: 服务基础 URL
        timeout: 超时时间（秒）
        api_key: 千帆 API key（qianfan 模式）
        pipeline_config: Pipeline 配置文件路径（local 模式）
        device: 设备（local 模式）

    Returns:
        Executor 实例
    """
    if source == "local":
        return LocalExecutor(
            pipeline=pipeline,
            pipeline_config=pipeline_config,
            device=device,
        )
    elif source == "aistudio":
        return AIStudioExecutor(
            pipeline=pipeline,
            token=token,
            base_url=base_url,
            request_timeout=float(timeout),
            poll_timeout=float(timeout * 10),
        )
    elif source == "qianfan":
        if not base_url:
            raise ValueError("base_url is required for qianfan source")
        if not api_key:
            raise ValueError("api_key is required for qianfan source")
        return QianfanExecutor(
            base_url=base_url,
            api_key=api_key,
            pipeline=pipeline,
            timeout=timeout,
        )
    elif source == "self_hosted":
        if not base_url:
            raise ValueError("base_url is required for self_hosted source")
        return SelfHostedExecutor(
            base_url=base_url,
            pipeline=pipeline,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown source: {source}")
```

- [ ] **Step 2: 更新 executors/__init__.py（移除重复的导入）**

```python
# mcp_server/paddleocr_mcp/executors/__init__.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base import Executor, ExecutorError, AuthenticationError, ResourceUnavailableError, TimeoutError
from .factory import create_executor

__all__ = [
    "Executor",
    "ExecutorError",
    "AuthenticationError",
    "ResourceUnavailableError",
    "TimeoutError",
    "create_executor",
]
```

- [ ] **Step 3: 提交**

```bash
cd /Users/linmanhui/Repos/PaddleOCR
git add mcp_server/paddleocr_mcp/executors/factory.py mcp_server/paddleocr_mcp/executors/__init__.py
git commit -m "refactor(mcp): add executor factory function"
```

---

## Task 7: 创建 capabilities 模块

**Files:**
- Create: `mcp_server/paddleocr_mcp/capabilities/__init__.py`
- Create: `mcp_server/paddleocr_mcp/capabilities/base.py`
- Create: `mcp_server/paddleocr_mcp/capabilities/ocr.py`
- Create: `mcp_server/paddleocr_mcp/capabilities/layout.py`
- Create: `mcp_server/paddleocr_mcp/capabilities/factory.py`

- [ ] **Step 1: 创建 capabilities/__init__.py**

```python
# mcp_server/paddleocr_mcp/capabilities/__init__.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base import MCPCapability
from .factory import create_capability

__all__ = [
    "MCPCapability",
    "create_capability",
]
```

- [ ] **Step 2: 创建 capabilities/base.py**

```python
# mcp_server/paddleocr_mcp/capabilities/base.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Any, Dict, List, Union

from fastmcp import Context
from mcp.types import ImageContent, TextContent

from ..executors.base import Executor


OutputMode = str  # Literal["simple", "detailed"]


class MCPCapability(abc.ABC):
    """MCP 能力抽象基类，负责工具注册和结果格式化"""

    def __init__(self, executor: Executor):
        self._executor = executor

    async def start(self) -> None:
        """启动能力"""
        await self._executor.start()

    async def stop(self) -> None:
        """停止能力"""
        await self._executor.stop()

    @abc.abstractmethod
    def register_tools(self, mcp: Any) -> None:
        """注册 MCP 工具

        Args:
            mcp: FastMCP 实例
        """
        pass

    @abc.abstractmethod
    def _format_result(
        self,
        result: Dict[str, Any],
        detailed: bool,
        **kwargs
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        """格式化结果为 MCP 输出

        Args:
            result: 统一格式的结果
            detailed: 是否使用详细格式
            **kwargs: 其他选项

        Returns:
            MCP 输出
        """
        pass

    async def _process(
        self,
        input_data: str,
        output_mode: str,
        ctx: Context,
        file_type: str = None,
        **options
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        """处理输入并格式化输出

        Args:
            input_data: 输入数据
            output_mode: 输出模式
            ctx: MCP 上下文
            file_type: 文件类型
            **options: 其他选项

        Returns:
            格式化后的输出
        """
        result = await self._executor.execute(input_data, file_type, **options)
        return self._format_result(result, output_mode == "detailed", **options)
```

- [ ] **Step 3: 创建 capabilities/ocr.py**

```python
# mcp_server/paddleocr_mcp/capabilities/ocr.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any, Dict, List, Union

from fastmcp import Context
from mcp.types import ImageContent, TextContent

from ..executors.base import Executor
from .base import MCPCapability


class OCRCapability(MCPCapability):
    """OCR MCP 能力"""

    def register_tools(self, mcp: Any) -> None:
        @mcp.tool("ocr")
        async def _ocr(
            input_data: str,
            output_mode: str = "simple",
            file_type: str = None,
            *,
            ctx: Context,
        ) -> Union[str, List[Union[TextContent, ImageContent]]]:
            """提取图像和 PDF 中的文字

            Args:
                input_data: 文件路径、URL 或 Base64 字符串
                output_mode: 输出模式
                    - "simple": 清晰可读的文本（默认）
                    - "detailed": 包含文本、置信度和边框坐标的 JSON
                file_type: 文件类型（URL 时必需）
                    - "image": 图像文件
                    - "pdf": PDF 文档
            """
            await ctx.info(f"--- OCR tool received `input_data`: {input_data[:50]} ---")
            return await self._process(input_data, output_mode, ctx, file_type)

    def _format_result(
        self,
        result: Dict[str, Any],
        detailed: bool,
        **kwargs
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        if not result["text"].strip():
            return (
                "❌ No text detected"
                if not detailed
                else json.dumps({"error": "No text detected"}, ensure_ascii=False)
            )

        if detailed:
            return json.dumps(result, ensure_ascii=False, indent=2)

        confidence = result["confidence"]
        text_line_count = len(result["text_lines"])

        output = result["text"]
        if confidence > 0:
            output += f"\n\n📊 Confidence: {(confidence * 100):.1f}% | {text_line_count} text lines"

        return output
```

- [ ] **Step 4: 创建 capabilities/layout.py**

```python
# mcp_server/paddleocr_mcp/capabilities/layout.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from typing import Any, Dict, List, Union

from fastmcp import Context
from mcp.types import ImageContent, TextContent

from ..executors.base import Executor
from .base import MCPCapability


class LayoutParsingCapability(MCPCapability):
    """版面解析 MCP 能力（PP-StructureV3 / PaddleOCR-VL）"""

    def __init__(self, executor: Executor, tool_name: str):
        """
        Args:
            executor: 执行器
            tool_name: MCP 工具名称
                - "pp_structurev3": 用于 PP-StructureV3 pipeline
                - "paddleocr_vl": 用于 PaddleOCR-VL 系列 pipeline
        """
        super().__init__(executor)
        self._tool_name = tool_name

    def register_tools(self, mcp: Any) -> None:
        @mcp.tool(self._tool_name)
        async def _layout_parsing(
            input_data: str,
            output_mode: str = "simple",
            file_type: str = None,
            return_images: bool = True,
            *,
            ctx: Context,
        ) -> Union[str, List[Union[TextContent, ImageContent]]]:
            """从复杂文档中提取结构化 Markdown

            Args:
                input_data: 文件路径、URL 或 Base64 字符串
                output_mode: 输出模式
                    - "simple": 清晰可读的 Markdown（默认）
                    - "detailed": JSON 格式包含版面信息
                file_type: 文件类型（URL 时必需）
                return_images: 是否返回提取的图片
            """
            return await self._process(
                input_data, output_mode, ctx, file_type, return_images=return_images
            )

    def _format_result(
        self,
        result: Dict[str, Any],
        detailed: bool,
        return_images: bool = True,
        **kwargs
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        if not result["markdown"].strip():
            return (
                "❌ No document content detected"
                if not detailed
                else json.dumps({"error": "No content detected"}, ensure_ascii=False)
            )

        markdown_text = result["markdown"]
        images_mapping = result.get("images_mapping", {})

        if return_images and images_mapping:
            content_list = self._parse_markdown_with_images(markdown_text, images_mapping)
        else:
            content_list = [TextContent(type="text", text=markdown_text)]

        if detailed:
            content_list.append(
                TextContent(
                    type="text",
                    text=f"Pages: {result['pages']}",
                )
            )

        if len(content_list) == 1:
            return content_list[0]
        return content_list

    def _parse_markdown_with_images(
        self, markdown_text: str, images_mapping: Dict[str, str]
    ) -> List[Union[TextContent, ImageContent]]:
        """解析 markdown 文本，返回混合的文本和图片内容"""
        if not images_mapping:
            return [TextContent(type="text", text=markdown_text)]

        content_list = []
        img_pattern = r'<img[^>]+src="([^"]+)"[^>]*>'
        last_pos = 0

        for match in re.finditer(img_pattern, markdown_text):
            text_before = markdown_text[last_pos:match.start()]
            if text_before.strip():
                content_list.append(TextContent(type="text", text=text_before))

            img_src = match.group(1)
            if img_src in images_mapping:
                content_list.append(
                    ImageContent(
                        type="image",
                        data=images_mapping[img_src],
                        mimeType="image/jpeg",
                    )
                )

            last_pos = match.end()

        remaining_text = markdown_text[last_pos:]
        if remaining_text.strip():
            content_list.append(TextContent(type="text", text=remaining_text))

        return content_list or [TextContent(type="text", text=markdown_text)]
```

- [ ] **Step 5: 创建 capabilities/factory.py**

```python
# mcp_server/paddleocr_mcp/capabilities/factory.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base import MCPCapability
from .ocr import OCRCapability
from .layout import LayoutParsingCapability


def create_capability(pipeline: str, executor) -> MCPCapability:
    """根据 pipeline 类型创建对应的 Capability

    Args:
        pipeline: Pipeline 类型
        executor: Executor 实例

    Returns:
        MCPCapability 实例
    """
    if pipeline == "OCR":
        return OCRCapability(executor)
    elif pipeline == "PP-StructureV3":
        return LayoutParsingCapability(executor, tool_name="pp_structurev3")
    elif pipeline in ("PaddleOCR-VL", "PaddleOCR-VL-1.5", "PaddleOCR-VL-1.6"):
        return LayoutParsingCapability(executor, tool_name="paddleocr_vl")
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")
```

- [ ] **Step 6: 提交**

```bash
cd /Users/linmanhui/Repos/PaddleOCR
git add mcp_server/paddleocr_mcp/capabilities/
git commit -m "refactor(mcp): add capabilities module with OCR and layout parsing"
```

---

## Task 8: 更新 __main__.py 使用新架构

**Files:**
- Modify: `mcp_server/paddleocr_mcp/__main__.py`

- [ ] **Step 1: 完全替换 __main__.py**

```python
#!/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import os
import sys

from fastmcp import FastMCP

from .executors import create_executor
from .capabilities import create_capability


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PaddleOCR MCP server - Supports local library, AI Studio service, and self-hosted servers."
    )

    parser.add_argument(
        "--pipeline",
        choices=[
            "OCR",
            "PP-StructureV3",
            "PaddleOCR-VL",
            "PaddleOCR-VL-1.5",
            "PaddleOCR-VL-1.6",
        ],
        default=os.getenv("PADDLEOCR_MCP_PIPELINE", "OCR"),
        help="Pipeline name.",
    )
    parser.add_argument(
        "--ppocr_source",
        choices=["local", "aistudio", "qianfan", "self_hosted"],
        default=os.getenv("PADDLEOCR_MCP_PPOCR_SOURCE", "local"),
        help="Source of PaddleOCR functionality: local (local library), aistudio (AI Studio service), qianfan (Qianfan service), self_hosted (self-hosted server).",
    )

    parser.add_argument(
        "--http",
        action="store_true",
        help="Use HTTP transport instead of STDIO (suitable for remote deployment and multiple clients).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address for HTTP mode (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP mode (default: 8000).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging for debugging."
    )

    # Local mode configuration
    parser.add_argument(
        "--pipeline_config",
        default=os.getenv("PADDLEOCR_MCP_PIPELINE_CONFIG"),
        help="PaddleOCR pipeline configuration file path (for local mode).",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("PADDLEOCR_MCP_DEVICE"),
        help="Device to run inference on.",
    )

    # Service mode configuration
    parser.add_argument(
        "--server_url",
        default=os.getenv("PADDLEOCR_MCP_SERVER_URL"),
        help="Base URL of the underlying service (required in qianfan/self_hosted mode).",
    )
    parser.add_argument(
        "--aistudio_access_token",
        default=os.getenv("PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN"),
        help="AI Studio access token (required for AI Studio).",
    )
    parser.add_argument(
        "--qianfan_api_key",
        default=os.getenv("PADDLEOCR_MCP_QIANFAN_API_KEY"),
        help="Qianfan API key (required for Qianfan).",
    )
    parser.add_argument(
        "--base_url",
        default=os.getenv("PADDLEOCR_MCP_BASE_URL"),
        help="Custom base URL for AI Studio (optional).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("PADDLEOCR_MCP_TIMEOUT", "60")),
        help="HTTP read timeout in seconds for API requests to the underlying server.",
    )

    args = parser.parse_args()
    return args


def _validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if not args.http and (args.host != "127.0.0.1" or args.port != 8000):
        print(
            "Host and port arguments are only valid when using HTTP transport (see: `--http`).",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.ppocr_source == "aistudio":
        if not args.aistudio_access_token:
            print("Error: The AI Studio access token is required.", file=sys.stderr)
            print(
                "Please either set `--aistudio_access_token` or set the environment variable "
                "`PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN`.",
                file=sys.stderr,
            )
            sys.exit(2)
    elif args.ppocr_source == "qianfan":
        if not args.server_url:
            print("Error: The server base URL is required.", file=sys.stderr)
            print(
                "Please either set `--server_url` or set the environment variable "
                "`PADDLEOCR_MCP_SERVER_URL`.",
                file=sys.stderr,
            )
            sys.exit(2)
        if not args.qianfan_api_key:
            print("Error: The Qianfan API key is required.", file=sys.stderr)
            print(
                "Please either set `--qianfan_api_key` or set the environment variable "
                "`PADDLEOCR_MCP_QIANFAN_API_KEY`.",
                file=sys.stderr,
            )
            sys.exit(2)
        if args.pipeline not in ("PP-StructureV3", "PaddleOCR-VL"):
            print(
                f"{repr(args.pipeline)} is currently not supported when using the {repr(args.ppocr_source)} source.",
                file=sys.stderr,
            )
            sys.exit(2)
    elif args.ppocr_source == "self_hosted":
        if not args.server_url:
            print("Error: The server base URL is required.", file=sys.stderr)
            print(
                "Please either set `--server_url` or set the environment variable "
                "`PADDLEOCR_MCP_SERVER_URL`.",
                file=sys.stderr,
            )
            sys.exit(2)


async def async_main() -> None:
    """Asynchronous main entry point."""
    args = _parse_args()
    _validate_args(args)

    # 创建 Executor
    executor = create_executor(
        source=args.ppocr_source,
        pipeline=args.pipeline,
        token=args.aistudio_access_token,
        base_url=args.base_url or args.server_url,
        timeout=args.timeout,
        api_key=args.qianfan_api_key,
        pipeline_config=args.pipeline_config,
        device=args.device,
    )

    # 创建 Capability
    capability = create_capability(
        pipeline=args.pipeline,
        executor=executor,
    )

    try:
        await capability.start()

        server_name = f"PaddleOCR {args.pipeline} MCP server"
        mcp = FastMCP(
            name=server_name,
            mask_error_details=True,
        )

        capability.register_tools(mcp)

        log_level = "INFO" if args.verbose else "WARNING"

        if args.http:
            await mcp.run_async(
                transport="streamable-http",
                host=args.host,
                port=args.port,
                log_level=log_level,
            )
        else:
            await mcp.run_async(log_level=log_level)

    except Exception as e:
        print(f"Failed to start the server: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    finally:
        await capability.stop()


def main():
    """Main entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
cd /Users/linmanhui/Repos/PaddleOCR
git add mcp_server/paddleocr_mcp/__main__.py
git commit -m "refactor(mcp): update __main__.py to use new architecture"
```

---

## Task 9: 更新版本号

**Files:**
- Modify: `mcp_server/pyproject.toml`

- [ ] **Step 1: 更新版本号到 0.7.0**

将 `version = "0.6.0"` 改为 `version = "0.7.0"`

- [ ] **Step 2: 提交**

```bash
cd /Users/linmanhui/Repos/PaddleOCR
git add mcp_server/pyproject.toml
git commit -m "chore(mcp): bump version to 0.7.0"
```

---

## Task 10: 删除旧代码

**Files:**
- Delete: `mcp_server/paddleocr_mcp/pipelines.py`

- [ ] **Step 1: 删除 pipelines.py**

```bash
cd /Users/linmanhui/Repos/PaddleOCR
rm mcp_server/paddleocr_mcp/pipelines.py
```

- [ ] **Step 2: 提交**

```bash
cd /Users/linmanhui/Repos/PaddleOCR
git add mcp_server/paddleocr_mcp/pipelines.py
git commit -m "refactor(mcp): remove old pipelines.py"
```

---

## Task 11: 验证安装

- [ ] **Step 1: 验证 import 是否正常**

```bash
cd /Users/linmanhui/Repos/PaddleOCR
python -c "from paddleocr_mcp.executors import create_executor; from paddleocr_mcp.capabilities import create_capability; print('Import successful')"
```

Expected: `Import successful`

- [ ] **Step 2: 验证 CLI 帮助信息**

```bash
cd /Users/linmanhui/Repos/PaddleOCR/mcp_server
python -m paddleocr_mcp --help
```

Expected: 显示帮助信息，包含 `--base_url` 参数

---

## Task 12: 手动测试（可选，需要真实 token）

- [ ] **Step 1: 测试 AI Studio 模式**

```bash
# 设置 token
export PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN="your-token"

# 启动 server（不会实际运行，只验证启动）
timeout 5 python -m paddleocr_mcp --pipeline OCR --ppocr_source aistudio || true
```

Expected: server 启动，5秒后超时退出（因为没有连接到真正的 MCP 客户端）

---

## 总结

此计划将现有的单一 `PipelineHandler` 架构重构为 `Executor` + `MCPCapability` 两层架构：

1. **Executor 层**：负责底层推理执行
   - `AIStudioExecutor`：使用 AsyncPaddleOCRClient（新功能）
   - `LocalExecutor`：本地推理
   - `QianfanExecutor` / `SelfHostedExecutor`：HTTP API 调用

2. **MCPCapability 层**：负责 MCP 工具注册和结果格式化
   - `OCRCapability`：OCR 工具
   - `LayoutParsingCapability`：版面解析工具

3. **工厂函数**：
   - `create_executor()`：根据 source 创建对应的 Executor
   - `create_capability()`：根据 pipeline 创建对应的 Capability

Breaking Changes:
- `PADDLEOCR_MCP_SERVER_URL` 不再用于 aistudio 模式
- 新增 `PADDLEOCR_MCP_BASE_URL`（可选）用于 aistudio 自定义地址
- 版本号：0.7.0
