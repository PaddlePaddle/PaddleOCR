# PaddleOCR MCP Server 架构重构设计

## 背景

PaddleOCR 3.6.0 新增了官方 API 的 Python SDK（`PaddleOCRClient` 和 `AsyncPaddleOCRClient`），用于调用异步 API。官方 API 以后不会再支持同步 API，只支持异步 API。

当前 MCP Server 仍然使用直接 HTTP 调用的方式访问同步 API，需要改造为使用 SDK。

## 目标

1. 基于 SDK 改造 MCP Server 的 aistudio 模式
2. 不考虑后向兼容
3. 保持代码结构优雅

## 设计决策

| 项目 | 决策 | 理由 |
|-----|------|------|
| SDK 客户端 | `AsyncPaddleOCRClient` | 与 FastMCP 的 async 工具函数更好地集成 |
| 改造范围 | 仅 aistudio 模式 | qianfan 和 self_hosted 保持现状 |
| 用户体验 | 阻塞式 | 与当前体验一致，SDK 内部完成 submit → poll → fetch |
| Token 环境变量 | `PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN` | 作用域清晰，命名规范 |
| Base URL 环境变量 | `PADDLEOCR_MCP_BASE_URL`（可选） | 默认使用 SDK 内置值，需要时可覆盖 |
| 架构模式 | Executor + MCPCapability 两层 | 职责清晰，易于扩展 |
| 版本号 | `0.7.0` | semver 0.x 可自由变更 |

## 架构设计

### 模块结构

```
paddleocr_mcp/
├── executors/
│   ├── __init__.py
│   ├── base.py              # Executor 抽象基类
│   ├── http.py              # HTTPExecutor 基类
│   ├── local.py             # LocalExecutor
│   ├── aistudio.py          # AIStudioExecutor
│   ├── qianfan.py           # QianfanExecutor
│   └── self_hosted.py       # SelfHostedExecutor
├── capabilities/
│   ├── __init__.py
│   ├── base.py              # MCPCapability 抽象基类
│   ├── ocr.py               # OCRCapability
│   └── layout.py            # LayoutParsingCapability
└── __main__.py
```

### 核心接口

#### Executor 抽象类

```python
class Executor(abc.ABC):
    """执行器抽象基类，负责底层推理执行"""

    @abc.abstractmethod
    async def execute(
        self,
        input_data: str,
        file_type: Optional[str] = None,
        **options
    ) -> Dict[str, Any]:
        """执行推理，返回统一格式的结果"""
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

#### MCPCapability 抽象类

```python
class MCPCapability(abc.ABC):
    """MCP 能力抽象基类，负责工具注册和结果格式化"""

    def __init__(self, executor: Executor):
        self._executor = executor

    async def start(self) -> None:
        await self._executor.start()

    async def stop(self) -> None:
        await self._executor.stop()

    @abc.abstractmethod
    def register_tools(self, mcp: FastMCP) -> None:
        """注册 MCP 工具"""
        pass

    @abc.abstractmethod
    def _format_result(
        self,
        result: Dict[str, Any],
        detailed: bool,
        **kwargs
    ) -> Union[str, List[Content]]:
        """格式化结果为 MCP 输出"""
        pass
```

### 数据流

```
用户调用 MCP 工具
    │
    ▼
MCPCapability.register_tools() 注册的工具函数
    │
    ├─ 参数验证、日志记录
    ├─ 调用 self._executor.execute()
    │   │
    │   ├─ AIStudioExecutor:
    │   │   ├─ 创建 AsyncPaddleOCRClient (start 时)
    │   │   ├─ 调用 client.ocr() / client.parse_document()
    │   │   ├─ 解析 SDK 结果为统一格式
    │   │   └─ 返回 Dict
    │   │
    │   ├─ LocalExecutor:
    │   │   ├─ 加载 PaddleOCR 模型 (start 时)
    │   │   ├─ 输入转换 (base64 → numpy)
    │   │   ├─ 调用 model.predict()
    │   │   └─ 解析结果为统一格式
    │   │
    │   └─ QianfanExecutor / SelfHostedExecutor (继承 HTTPExecutor):
    │       ├─ HTTP POST
    │       └─ 解析响应为统一格式
    │
    ├─ 调用 self._format_result()
    │   └─ 格式化为 MCP 输出
    │
    └─ 返回结果给用户
```

### 统一结果格式

#### OCR 结果格式

```python
{
    "text": "识别的文字",
    "confidence": 0.95,
    "text_lines": [
        {"text": "第一行", "confidence": 0.98, "bbox": [[x1,y1], ...]},
        ...
    ]
}
```

#### 版面解析结果格式

```python
{
    "markdown": "# 标题\n\n内容...",
    "pages": 3,
    "images_mapping": {"img1": "base64...", "img2": "base64..."},
    "detailed_results": [...]  # 可选，仅在 detailed 模式返回
}
```

### 命名约定

- `LocalExecutor`：本地推理执行器
- `AIStudioExecutor`：AI Studio 官方 API 执行器（使用 SDK）
- `QianfanExecutor`：千帆平台执行器（继承 HTTPExecutor）
- `SelfHostedExecutor`：自托管服务执行器（继承 HTTPExecutor）
- `HTTPExecutor`：同步 HTTP API 的抽象基类

命名侧重于"来源"（source）而非"实现细节"。

### Executor 实现

#### AIStudioExecutor

```python
class AIStudioExecutor(Executor):
    """使用 PaddleOCR SDK 的 AI Studio 执行器"""

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

    async def start(self) -> None:
        self._client = AsyncPaddleOCRClient(
            token=self._token,
            base_url=self._base_url,
            request_timeout=self._request_timeout,
            poll_timeout=self._poll_timeout,
        )
        await self._client._http.__aenter__()

    async def stop(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def execute(
        self,
        input_data: str,
        file_type: Optional[str] = None,
        **options
    ) -> Dict[str, Any]:
        # 根据 pipeline 类型调用不同的 SDK 方法
        # 解析 SDK 返回的 OCRResult / DocParsingResult
        # 转换为统一格式
        pass
```

#### LocalExecutor

```python
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
        self._engine = self._create_engine()
        self._engine_wrapper = _EngineWrapper(self._engine)

    async def stop(self) -> None:
        if self._engine_wrapper:
            await self._engine_wrapper.close()

    async def execute(
        self,
        input_data: str,
        file_type: Optional[str] = None,
        **options
    ) -> Dict[str, Any]:
        # 处理输入（可能需要转换为 numpy）
        # 调用 self._engine_wrapper.call()
        # 解析结果为统一格式
        pass
```

#### HTTPExecutor

```python
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
        response = await self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

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

#### QianfanExecutor

```python
class QianfanExecutor(HTTPExecutor):
    """千帆平台执行器"""

    def __init__(self, base_url: str, api_key: str, timeout: int = 60):
        super().__init__(base_url, timeout)
        self._api_key = api_key

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _get_endpoint(self) -> str:
        # 根据 pipeline 返回不同端点
        # OCR: "paddleocr"
        # Layout: "paddleocr"
        return self._pipeline_endpoint

    def _prepare_payload(self, input_data: str, file_type: Optional[str], **options) -> Dict[str, Any]:
        # 千帆特定的 payload 格式
        pass

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        # 解析千帆响应
        pass
```

#### SelfHostedExecutor

```python
class SelfHostedExecutor(HTTPExecutor):
    """自托管服务执行器"""

    def _get_headers(self) -> Dict[str, str]:
        return {}  # 无需认证

    def _get_endpoint(self) -> str:
        # OCR: "ocr"
        # Layout: "layout-parsing"
        return self._pipeline_endpoint

    def _prepare_payload(self, input_data: str, file_type: Optional[str], **options) -> Dict[str, Any]:
        # 自托管特定的 payload 格式
        pass

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        # 解析自托管响应
        pass
```

### Executor 创建工厂

```python
# executors/__init__.py

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
    """根据 source 创建对应的 Executor"""
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
            poll_timeout=float(timeout * 10),  # 轮询时间更长
        )
    elif source == "qianfan":
        return QianfanExecutor(
            base_url=base_url or "",
            api_key=api_key or "",
            timeout=timeout,
        )
    elif source == "self_hosted":
        return SelfHostedExecutor(
            base_url=base_url or "",
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown source: {source}")
```

### __main__.py 更新

```python
# __main__.py

from paddleocr_mcp.executors import create_executor
from paddleocr_mcp.capabilities import create_capability

async def async_main() -> None:
    args = _parse_args()
    _validate_args(args)

    # 创建 Executor
    executor = create_executor(
        source=args.ppocr_source,
        pipeline=args.pipeline,
        token=args.aistudio_access_token,
        base_url=args.base_url,
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
        mcp = FastMCP(name=server_name, mask_error_details=True)

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

    finally:
        await capability.stop()
```

### MCPCapability 创建工厂

```python
# capabilities/__init__.py

def create_capability(pipeline: str, executor: Executor) -> MCPCapability:
    """根据 pipeline 类型创建对应的 Capability"""
    if pipeline == "OCR":
        return OCRCapability(executor)
    elif pipeline == "PP-StructureV3":
        return LayoutParsingCapability(executor, tool_name="pp_structurev3")
    elif pipeline in ("PaddleOCR-VL", "PaddleOCR-VL-1.5", "PaddleOCR-VL-1.6"):
        return LayoutParsingCapability(executor, tool_name="paddleocr_vl")
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")
```

## MCP 能力与 Pipeline 映射

| Pipeline 类型 | Capability 类 | MCP 工具名 |
|--------------|--------------|-----------|
| OCR | `OCRCapability` | `ocr` |
| PP-StructureV3 | `LayoutParsingCapability` | `pp_structurev3` |
| PaddleOCR-VL | `LayoutParsingCapability` | `paddleocr_vl` |
| PaddleOCR-VL-1.5 | `LayoutParsingCapability` | `paddleocr_vl` |
| PaddleOCR-VL-1.6 | `LayoutParsingCapability` | `paddleocr_vl` |

#### OCRCapability

```python
class OCRCapability(MCPCapability):
    """OCR MCP 能力"""

    def register_tools(self, mcp: FastMCP) -> None:
        @mcp.tool("ocr")
        async def _ocr(
            input_data: str,
            output_mode: OutputMode = "simple",
            file_type: Optional[str] = None,
            *,
            ctx: Context,
        ) -> Union[str, List[Union[TextContent, ImageContent]]]:
            """提取图像和 PDF 中的文字"""
            return await self._process(input_data, output_mode, ctx, file_type)

    def _format_result(
        self,
        result: Dict[str, Any],
        detailed: bool,
        **kwargs
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        if not result["text"].strip():
            return "❌ No text detected" if not detailed else json.dumps({"error": "No text detected"})

        if detailed:
            return json.dumps(result, ensure_ascii=False, indent=2)

        confidence = result["confidence"]
        text_line_count = len(result["text_lines"])
        output = result["text"]
        if confidence > 0:
            output += f"\n\n📊 Confidence: {(confidence * 100):.1f}% | {text_line_count} text lines"
        return output
```

#### LayoutParsingCapability

```python
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

    def register_tools(self, mcp: FastMCP) -> None:
        @mcp.tool(self._tool_name)  # "pp_structurev3" 或 "paddleocr_vl"
        async def _layout_parsing(
            input_data: str,
            output_mode: OutputMode = "simple",
            file_type: Optional[str] = None,
            return_images: bool = True,
            *,
            ctx: Context,
        ) -> Union[str, List[Union[TextContent, ImageContent]]]:
            """从复杂文档中提取结构化 Markdown"""
            return await self._process(input_data, output_mode, ctx, file_type, return_images=return_images)

    def _format_result(
        self,
        result: Dict[str, Any],
        detailed: bool,
        return_images: bool = True,
        **kwargs
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        # 格式化 Markdown 输出，可选包含图片
        pass
```

### 错误处理

### 异常类型

```python
class ExecutorError(RuntimeError):
    """Executor 执行错误基类"""
    pass

class AuthenticationError(ExecutorError):
    """认证失败 (AIStudioExecutor)"""
    pass

class ResourceUnavailableError(ExecutorError):
    """服务不可用"""
    pass

class TimeoutError(ExecutorError):
    """超时"""
    pass
```

### Executor 错误映射

| Executor | 可能的 SDK/API 错误 | 转换为 ExecutorError |
|----------|-------------------|---------------------|
| AIStudioExecutor | `AuthError` | `AuthenticationError` |
| AIStudioExecutor | `ServiceUnavailableError` | `ResourceUnavailableError` |
| AIStudioExecutor | `JobFailedError`, `APIError` | `ExecutorError` |
| QianfanExecutor / SelfHostedExecutor | HTTP 401 | `AuthenticationError` |
| QianfanExecutor / SelfHostedExecutor | HTTP 503 | `ResourceUnavailableError` |
| QianfanExecutor / SelfHostedExecutor | 其他 HTTP 错误 | `ExecutorError` |
| LocalExecutor | 推理失败 | `ExecutorError` |

### 错误传播

```
SDK/API 错误
    │
    ▼
Executor 捕获并转换为 ExecutorError 子类
    │
    ▼
MCPCapability 捕获
    ├─ 记录错误日志 (ctx.error())
    └─ 重新抛出或返回错误消息
    │
    ▼
MCP 工具返回错误或抛出异常
```

## 配置变更

### CLI 参数

| 参数 | 类型 | 说明 |
|-----|------|------|
| `--ppocr_source` | str | `local`, `aistudio`, `qianfan`, `self_hosted` |
| `--pipeline` | str | `OCR`, `PP-StructureV3`, `PaddleOCR-VL`, `PaddleOCR-VL-1.5`, `PaddleOCR-VL-1.6` |
| `--base_url` | str | aistudio 模式的自定义 base_url |
| `--access_token` | str | aistudio 模式的 token |

### 环境变量

| 环境变量 | CLI 参数 | 说明 |
|---------|---------|------|
| `PADDLEOCR_MCP_BASE_URL` | `--base_url` | aistudio base_url（可选） |
| `PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN` | `--access_token` | aistudio token |
| `PADDLEOCR_MCP_SERVER_URL` | `--server_url` | qianfan/self_hosted（保持现状） |
| `PADDLEOCR_MCP_QIANFAN_API_KEY` | `--qianfan_api_key` | qianfan（保持现状） |

## 版本更新

**版本号：** `0.7.0`

**Breaking Changes：**
- `PADDLEOCR_MCP_SERVER_URL` 不再用于 aistudio 模式，改用 `PADDLEOCR_MCP_BASE_URL`（可选）
- 移除旧的环境变量支持（如有）

## 测试策略

1. **单元测试**：
   - 每个 Executor 的 `execute()` 方法
   - 结果格式转换逻辑
   - 错误处理逻辑

2. **集成测试**：
   - AIStudioSDKExecutor 与真实 SDK 的交互
   - MCPCapability 与 MCP 工具的集成

3. **端到端测试**：
   - 启动 MCP server，调用工具验证返回结果

## 迁移指南

### 从 0.6.0 迁移到 0.7.0

对于 aistudio 模式用户：

**之前：**
```json
{
  "env": {
    "PADDLEOCR_MCP_SERVER_URL": "https://xxxxxx.aistudio-app.com"
  }
}
```

**之后：**
```json
{
  "env": {
    "PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN": "your-token"
  }
}
```

如果需要使用自定义 base_url：
```json
{
  "env": {
    "PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN": "your-token",
    "PADDLEOCR_MCP_BASE_URL": "https://your-proxy.com"
  }
}
```
