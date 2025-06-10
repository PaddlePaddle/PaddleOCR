# 标准库导入
import logging
import os

# 第三方库导入
from fastmcp import Context

# 本地应用导入
from .pipelines import OcrPipeline, StructurePipeline

# 本地OCR库（可选导入）
try:
    from paddleocr import PaddleOCR, PPStructureV3

    LOCAL_OCR_AVAILABLE = True
except ImportError:
    LOCAL_OCR_AVAILABLE = False
    # 智能日志：只在可能需要本地模式时才警告
    ocr_source = os.getenv("PADDLEOCR_MCP_OCR_SOURCE", "local")
    if ocr_source == "local":
        # 本地模式：PaddleOCR缺失是问题，需要警告
        logging.getLogger(__name__).warning(
            "PaddleOCR not available. Local mode will be disabled. Install with: pip install paddleocr"
        )
    else:
        # API模式：PaddleOCR缺失是预期的，只需info级别
        logging.getLogger(__name__).info(
            "PaddleOCR not available. Running in API-only mode."
        )


# ==================== 统一引擎管理容器 ====================


class EngineContainer:
    """智能引擎容器 - 统一管理所有OCR产线"""

    def __init__(self):
        self._engines = {}
        self._api_config = None
        self._engine_configs = {}
        self.SUPPORTED_ENGINES = ["ocr", "structure"]

    def configure_api(
        self, api_url: str, service_type: str, api_token: str = None, timeout: int = 30
    ):
        """配置API连接参数 - 明确指定服务类型"""
        self._api_config = {
            "url": api_url,
            "token": api_token,
            "timeout": timeout,
            "is_user_service": service_type == "user_service",
            "service_type": (
                "layout-parsing" if "layout-parsing" in api_url.lower() else "ocr"
            ),
        }

    def configure_local_engines(
        self, ocr_config_path: str = None, structure_config_path: str = None
    ):
        """配置本地引擎YAML配置文件路径"""
        if not ocr_config_path:
            ocr_config_path = os.path.join(
                os.path.dirname(__file__), "..", "configs", "OCR.yaml"
            )
        if not structure_config_path:
            structure_config_path = os.path.join(
                os.path.dirname(__file__), "..", "configs", "PP-StructureV3.yaml"
            )

        self._engine_configs = {
            "ocr": ocr_config_path,
            "structure": structure_config_path,
        }

    def is_api_mode(self) -> bool:
        """判断是否为API模式"""
        return self._api_config is not None

    def get_engine(self, engine_name: str):
        """智能获取引擎实例 - 支持按需或预初始化"""
        if engine_name not in self._engines:
            self._engines[engine_name] = self._create_engine(engine_name)
        return self._engines[engine_name]

    def _create_engine(self, engine_name: str):
        """创建指定引擎实例 - 直接使用官方YAML配置"""
        if not LOCAL_OCR_AVAILABLE:
            raise RuntimeError("PaddleOCR not available. Please install paddleocr.")

        config_path = self._engine_configs.get(engine_name)
        if not config_path or not os.path.exists(config_path):
            raise ValueError(f"Configuration file not found for engine: {engine_name}")

        if engine_name == "ocr":
            return PaddleOCR(paddlex_config=config_path)
        elif engine_name == "structure":
            return PPStructureV3(paddlex_config=config_path)
        else:
            raise ValueError(f"Unknown engine: {engine_name}")

    def warmup_engines(self, engine_types: list = None):
        """预热引擎 - eager初始化，避免首次运行延迟"""
        if not LOCAL_OCR_AVAILABLE:
            return

        for engine_type in engine_types or self.SUPPORTED_ENGINES:
            try:
                self.get_engine(engine_type)
            except Exception:
                pass  # 静默失败，不影响启动


# 全局引擎容器实例
engines = EngineContainer()


# ==================== MCP工具注册 ====================


def register_tools(
    mcp,
    ocr_source_type: str = "local",
    tool_type: str = "auto",
    ocr_config_path: str = None,
    structure_config_path: str = None,
    **api_config,
):
    """注册MCP工具 - 智能工具注册策略

    工具注册规则：
    • 本地模式(local): 注册两个工具（ocr_text + ocr_structure）
    • API模式(aistudio/user_service): 每个服务注册一个专用工具，根据URL自动判断

    Args:
        mcp: FastMCP实例
        ocr_source_type: 数据源类型 ("local", "aistudio", "user_service")
        tool_type: 工具类型 ("auto", "ocr", "structure") - API模式使用
        ocr_config_path: OCR产线YAML配置文件路径（本地模式）
        structure_config_path: 结构分析产线YAML配置文件路径（本地模式）
        **api_config: API配置参数（必须包含api_url）
    """

    # 配置数据源
    if ocr_source_type in ["aistudio", "user_service"]:
        engines.configure_api(service_type=ocr_source_type, **api_config)
    else:
        engines.configure_local_engines(ocr_config_path, structure_config_path)
        engines.warmup_engines()

    # 智能描述
    if engines.is_api_mode():
        source_desc = (
            "用户服务API" if engines._api_config["is_user_service"] else "星河API"
        )
    else:
        source_desc = "本地PaddleOCR"

    # 确定要注册的工具
    if ocr_source_type == "local":
        # 本地模式：注册两个工具（支持完整功能）
        tools_to_register = ["ocr", "structure"]
    else:
        # API模式（星河API + 用户服务API）：每个服务专注一个工具，根据URL判断
        if tool_type == "auto":
            # 智能识别：从URL推断工具类型
            api_url = api_config.get("api_url", "").lower()
            if "layout-parsing" in api_url or "structure" in api_url:
                tools_to_register = ["structure"]
            else:
                tools_to_register = ["ocr"]
        else:
            # 明确指定工具类型
            tools_to_register = [tool_type]

    # 创建产线处理器实例
    ocr_pipeline = OcrPipeline(engines)
    structure_pipeline = StructurePipeline(engines)

    # 注册OCR工具
    if "ocr" in tools_to_register:

        @mcp.tool()
        async def ocr_text(
            input_data: str,
            output_mode: str = "simple",
            ctx: Context = None,
        ) -> str:
            f"""Extract text from images and PDFs using {source_desc}

            Args:
                input_data: File path, URL, or base64 data
                output_mode: "simple" for clean text, "detailed" for JSON with positioning
            """
            return await ocr_pipeline.process(input_data, output_mode, ctx)

    # 注册结构分析工具
    if "structure" in tools_to_register:

        @mcp.tool()
        async def ocr_structure(
            input_data: str,
            output_mode: str = "simple",
            ctx: Context = None,
        ):
            f"""Extract document structure and layout using {source_desc}

            Args:
                input_data: File path, URL, or base64 data
                output_mode: "simple" for markdown, "detailed" for JSON with metadata

            Returns: Markdown text + images (if available) or structured JSON
            """
            return await structure_pipeline.process(input_data, output_mode, ctx)
