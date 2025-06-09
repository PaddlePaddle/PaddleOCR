# 标准库导入
import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

# 第三方库导入
import numpy as np
import httpx
from PIL import Image
from fastmcp import Context, Image as FastMCPImage

# 配置logger - 模块错误
logger = logging.getLogger(__name__)

# 本地OCR库（可选导入）
try:
    from paddleocr import PaddleOCR, PPStructureV3

    LOCAL_OCR_AVAILABLE = True
except ImportError:
    LOCAL_OCR_AVAILABLE = False
    logger.warning(
        "PaddleOCR not available. Local mode will be disabled. Install with: pip install paddleocr"
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

    def load_config(self, config_file: str = None):
        """加载引擎配置文件"""
        # 默认使用当前目录的local_config.json
        if not config_file:
            config_file = os.path.join(os.path.dirname(__file__), "local_config.json")

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                self._engine_configs = config.get("engines", {})
                logger.info(f"Loaded engine config from {config_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config file {config_file}: {e}")

    def is_api_mode(self) -> bool:
        """判断是否为API模式"""
        return self._api_config is not None

    def get_engine(self, engine_name: str):
        """智能获取引擎实例 - 支持按需或预初始化"""
        if engine_name not in self._engines:
            self._engines[engine_name] = self._create_engine(engine_name)
        return self._engines[engine_name]

    def _create_engine(self, engine_name: str):
        """创建指定引擎实例"""
        if not LOCAL_OCR_AVAILABLE:
            raise RuntimeError("PaddleOCR not available. Please install paddleocr.")

        # 直接使用配置文件中的参数
        config = self._engine_configs.get(engine_name, {})
        if not config:
            raise ValueError(f"No configuration found for engine: {engine_name}")

        if engine_name == "ocr":
            return PaddleOCR(**config)
        elif engine_name == "structure":
            return PPStructureV3(**config)
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

# ==================== 输入处理 ====================


def _is_file_path(data: str) -> bool:
    """简单文件路径判断"""
    return data.startswith(("/", "./", "../")) or "\\" in data


def _is_base64(data: str) -> bool:
    """简单base64判断"""
    return (
        len(data) > 100
        and data.replace("+", "").replace("/", "").replace("=", "").isalnum()
    )


def _detect_file_type(input_data: str) -> int:
    """检测文件类型：0=PDF，1=图片"""
    # 文件路径：检查扩展名
    if _is_file_path(input_data):
        return 0 if input_data.lower().endswith(".pdf") else 1

    # Base64：检查PDF魔术字节
    if _is_base64(input_data) and input_data.startswith("JVBERi"):
        return 0

    # 默认图片
    return 1


def _process_for_local(input_data: str):
    """本地模式输入处理：base64转numpy，文件路径直接用"""
    if _is_base64(input_data):
        if input_data.startswith("data:"):
            base64_data = input_data.split(",", 1)[1]
        else:
            base64_data = input_data
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes))
        return np.array(image)
    return input_data


# ==================== API调用 ====================


async def _call_api(input_data: str) -> dict:
    """调用API - 支持星河API和用户服务API"""
    if not engines._api_config:
        raise ValueError("API not configured")

    # 文件路径 -> base64，其他原样传递
    if _is_file_path(input_data):
        with open(input_data, "rb") as f:
            file_data = base64.b64encode(f.read()).decode("ascii")
    else:
        file_data = input_data

    # API需要文件类型参数
    payload = {"file": file_data, "fileType": _detect_file_type(input_data)}

    # 构建headers
    headers = {"Content-Type": "application/json"}
    if engines._api_config["token"]:
        headers["Authorization"] = f'token {engines._api_config["token"]}'

    # 发送请求
    async with httpx.AsyncClient() as client:
        response = await client.post(
            engines._api_config["url"],
            headers=headers,
            json=payload,
            timeout=engines._api_config.get("timeout", 30),
        )

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    response_json = response.json()
    error_code = response_json.get("errorCode", 0)

    if error_code != 0:
        error_msg = response_json.get("errorMsg", "Unknown API error")
        raise RuntimeError(f"API failed (errorCode: {error_code}): {error_msg}")

    return response_json["result"]


# ==================== 结果解析 ====================


def _parse_ocr_result(raw_result) -> dict:
    """解析OCR结果 - 直接返回统一格式"""
    if engines.is_api_mode():
        return _parse_api_ocr(raw_result)
    else:
        return _parse_local_ocr(raw_result)


def _parse_api_ocr(api_result: dict) -> dict:
    """解析星河API OCR结果 - 直接构建最终格式"""
    ocr_results = api_result["ocrResults"]
    if not ocr_results:
        return {
            "text": "",
            "confidence": 0,
            "blocks": [],
            "text_type": "api",
            "det_params": None,
        }

    # 直接提取和组装
    all_texts, all_confidences, blocks = [], [], []

    for ocr_result in ocr_results:
        pruned = ocr_result["prunedResult"]

        # prunedResult确定是字典类型
        texts = pruned.get("rec_texts", [])
        scores = pruned.get("rec_scores", [])
        boxes = pruned.get("rec_boxes", [])

        for i, text in enumerate(texts):
            if text and text.strip():
                conf = scores[i] if i < len(scores) else 0
                all_texts.append(text.strip())
                all_confidences.append(conf)

                block = {"text": text.strip(), "confidence": round(conf, 3)}
                if i < len(boxes) and boxes[i]:
                    block["bbox"] = boxes[i]
                blocks.append(block)

    return {
        "text": "\n".join(all_texts),
        "confidence": (
            sum(all_confidences) / len(all_confidences) if all_confidences else 0
        ),
        "blocks": blocks,
        "text_type": "api",
        "det_params": None,
    }


def _parse_local_ocr(raw_result) -> dict:
    """解析本地OCR结果 - 直接构建最终格式"""
    if not raw_result or not raw_result[0]:
        return {
            "text": "",
            "confidence": 0,
            "blocks": [],
            "text_type": "local",
            "det_params": None,
        }

    result = raw_result[0]
    texts = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])
    boxes = result.get("rec_boxes", []) or result.get("rec_polys", [])

    if not texts:
        return {
            "text": "",
            "confidence": 0,
            "blocks": [],
            "text_type": "local",
            "det_params": None,
        }

    # 直接组装
    clean_texts, confidences, blocks = [], [], []

    for i, text in enumerate(texts):
        if text and text.strip():
            conf = scores[i] if i < len(scores) else 0
            clean_texts.append(text.strip())
            confidences.append(conf)

            block = {"text": text.strip(), "confidence": round(conf, 3)}
            if i < len(boxes) and boxes[i]:
                block["bbox"] = boxes[i]
            blocks.append(block)

    return {
        "text": "\n".join(clean_texts),
        "confidence": sum(confidences) / len(confidences) if confidences else 0,
        "blocks": blocks,
        "text_type": result.get("text_type", "local"),
        "det_params": result.get("text_det_params"),
    }


def _parse_structure_result(raw_result) -> dict:
    """解析结构分析结果 - 直接返回统一格式"""
    if engines.is_api_mode():
        return _parse_api_structure(raw_result)
    else:
        return _parse_local_structure(raw_result)


def _parse_api_structure(api_result: dict) -> dict:
    """解析星河API结构结果 - 直接构建最终格式"""
    layout_results = api_result["layoutParsingResults"]
    if not layout_results:
        return {
            "markdown": "",
            "pages": [],
            "has_images": False,
            "images": [],
            "referenced_images": [],
        }

    markdown_parts, pages, all_images = [], [], []
    referenced_images = []  # 所有在markdown中被引用的图片

    for i, res in enumerate(layout_results):
        markdown_data = res["markdown"]  # 直接访问，因为markdown是必须字段
        page_images = []

        if markdown_data.get("text"):
            text = markdown_data["text"]
            markdown_parts.append(text)

            # 收集当前页面的图片
            if markdown_data.get("images"):
                sorted_images = sorted(markdown_data["images"].items())
                page_images = [url for filename, url in sorted_images]
                all_images.extend(page_images)

                # 提取当前页面markdown中实际引用的所有图片
                page_referenced_images = _extract_referenced_images_from_markdown(
                    text, markdown_data["images"]
                )
                referenced_images.extend(page_referenced_images)

            page = {
                "page": i,
                "content": text,
                "has_images": bool(page_images),
                "images": page_images,  # 保存页面级图片关联
            }
            pages.append(page)

    return {
        "markdown": "\n".join(markdown_parts),
        "pages": pages,
        "has_images": bool(all_images),
        "images": all_images,
        "referenced_images": referenced_images,  # 所有在markdown中被引用的图片
    }


def _parse_local_structure(raw_results) -> dict:
    """解析本地结构结果 - 直接构建最终格式"""
    if not raw_results:
        return {"markdown": "", "pages": [], "has_images": False, "images": []}

    markdown_parts, pages = [], []

    for i, result in enumerate(raw_results):
        text = ""
        if hasattr(result, "markdown") and result.markdown:
            text = (
                result.markdown.get("text", str(result.markdown))
                if isinstance(result.markdown, dict)
                else str(result.markdown)
            )

        if text:
            markdown_parts.append(text)
            pages.append({"page": i, "content": text, "has_images": False})

    return {
        "markdown": "\n".join(markdown_parts),
        "pages": pages,
        "has_images": False,
        "images": [],
    }


# ==================== 格式化输出 ====================


def format_ocr_output(result: dict, detailed: bool = False) -> str:
    """格式化OCR输出 - L1核心信息，L2完整数据"""
    if not result["text"].strip():
        return (
            "❌ No text detected"
            if not detailed
            else json.dumps({"error": "No text detected"}, ensure_ascii=False)
        )

    if detailed:
        # L2: 直接返回解析结果，无包装
        return json.dumps(result, ensure_ascii=False, indent=2)
    else:
        # L1: 核心文本 + 关键统计
        confidence = result["confidence"]
        block_count = len(result["blocks"])

        output = result["text"]
        if confidence > 0:
            output += (
                f"\n\n📊 置信度: {(confidence * 100):.1f}% | {block_count}个文本块"
            )

        return output


def format_structure_output(
    result: dict, detailed: bool = False, include_image_refs: bool = True
) -> str:
    """格式化结构分析输出 - L1/L2都包含图片"""
    if not result["markdown"].strip():
        return (
            "❌ No document structure detected"
            if not detailed
            else json.dumps({"error": "No structure detected"}, ensure_ascii=False)
        )

    if detailed:
        # L2: 完整数据直接返回
        return json.dumps(result, ensure_ascii=False, indent=2)
    else:
        # L1: 纯markdown + 图片引用（如果有）
        markdown = result["markdown"]

        if result["images"] and include_image_refs:
            image_refs = "\n\n📸 **Images**: " + ", ".join(
                f"[img{i+1}]({url})" for i, url in enumerate(result["images"])
            )
            markdown += image_refs

        return markdown


# ==================== 图片引用解析 ====================


def _extract_referenced_images_from_markdown(
    markdown_text: str, available_images: Dict[str, str]
) -> List[str]:
    """从markdown文本中提取实际被引用的图片URL"""
    if not markdown_text or not available_images:
        return []

    # 匹配markdown中的图片引用：<img src="path" /> 或 ![alt](path)
    img_patterns = [
        r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>',  # HTML img标签
        r"!\[[^\]]*\]\(([^)]+)\)",  # Markdown图片语法
    ]

    referenced_images = []
    for pattern in img_patterns:
        matches = re.findall(pattern, markdown_text)
        for img_path in matches:
            # 提取文件名（去掉路径前缀）
            img_filename = img_path.split("/")[-1]
            # 在available_images中查找对应的URL
            for filename, url in available_images.items():
                if filename == img_filename or img_path.endswith(filename):
                    referenced_images.append(url)
                    break

    return referenced_images


# ==================== MCP工具注册 ====================


def register_tools(
    mcp,
    ocr_source_type: str = "local",
    tool_type: str = "auto",
    config_file: str = None,
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
        config_file: 配置文件路径（本地模式可选）
        **api_config: API配置参数（必须包含api_url）
    """

    # 配置数据源
    if ocr_source_type in ["aistudio", "user_service"]:
        engines.configure_api(service_type=ocr_source_type, **api_config)
    else:
        engines.load_config(config_file)  # 加载本地模式配置
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

    # 注册OCR工具
    if "ocr" in tools_to_register:

        @mcp.tool()
        async def ocr_text(
            input_data: str,
            output_mode: str = "simple",
            ctx: Optional[Context] = None,
        ) -> str:
            f"""🔍 Extract text from images and PDFs - **Supports URLs, file paths, and base64 data**, for most cases, use "simple" mode is enough. details mode contain position layout information, which is not necessary for most cases.

            Args:
                input_data: File path, URL, or base64 data
                output_mode: "simple" for clean text, "detailed" for JSON with positioning

            Powered by: {source_desc}
            """
            try:
                if ctx:
                    await ctx.info(f"Starting OCR processing using {source_desc}")

                if engines.is_api_mode():
                    if not ctx:
                        raise ValueError("Context required for API mode")
                    raw_result = await _call_api(input_data)
                else:
                    # 本地模式：base64转numpy，文件路径直接用
                    processed_input = _process_for_local(input_data)

                    ocr = engines.get_engine("ocr")
                    loop = asyncio.get_running_loop()
                    raw_result = await loop.run_in_executor(
                        None, ocr.predict, processed_input
                    )

                result = _parse_ocr_result(raw_result)

                if ctx:
                    text_length = len(result["text"])
                    block_count = len(result["blocks"])
                    await ctx.info(
                        f"OCR completed: {text_length} characters, {block_count} text blocks"
                    )

                return format_ocr_output(result, output_mode == "detailed")

            except Exception as e:
                if ctx:
                    await ctx.error(f"OCR failed: {str(e)}")
                error_msg = f"OCR failed: {str(e)}"
                return (
                    error_msg
                    if output_mode == "simple"
                    else json.dumps({"error": error_msg}, ensure_ascii=False)
                )

    # 注册结构分析工具
    if "structure" in tools_to_register:

        @mcp.tool()
        async def ocr_structure(
            input_data: str,
            output_mode: str = "simple",
            ctx: Optional[Context] = None,
        ):
            f"""🏗️ Extract document structure and layout - **Supports URLs, file paths, and base64 data**, for most cases, use "simple" mode is enough. details mode contain position layout information, which is not necessary for most cases.

            Args:
                input_data: File path, URL, or base64 data
                output_mode: "simple" for markdown, "detailed" for JSON with metadata

            Returns: Markdown text + images (if available) or structured JSON
            Powered by: {source_desc}
            """
            try:
                if ctx:
                    await ctx.info(f"Starting structure analysis using {source_desc}")

                if engines.is_api_mode():
                    if not ctx:
                        raise ValueError("Context required for API mode")
                    raw_result = await _call_api(input_data)
                else:
                    # 本地模式：base64转numpy，文件路径直接用
                    processed_input = _process_for_local(input_data)

                    structure = engines.get_engine("structure")
                    loop = asyncio.get_running_loop()
                    raw_result = await loop.run_in_executor(
                        None, structure.predict, processed_input
                    )

                result = _parse_structure_result(raw_result)

                if ctx:
                    page_count = len(result["pages"])
                    image_count = len(result["images"])
                    await ctx.info(
                        f"Structure analysis completed: {page_count} pages, {image_count} images"
                    )

                # 🖼️ 混合内容传输
                if result["images"] and engines.is_api_mode():
                    try:
                        import base64

                        content_list = []

                        # 添加文本内容（无图片引用避免重复）
                        text_content = format_structure_output(
                            result, output_mode == "detailed", include_image_refs=False
                        )
                        content_list.append(text_content)

                        # 添加所有在markdown中引用的图片
                        referenced_images = result.get("referenced_images", [])
                        for target_image in referenced_images:
                            # 检测是否为URL
                            if target_image.startswith(("http://", "https://")):
                                # URL：下载图片
                                async with httpx.AsyncClient() as client:
                                    try:
                                        response = await client.get(target_image)
                                        if response.status_code == 200:
                                            image_format = response.headers.get(
                                                "content-type", "image/jpeg"
                                            ).split("/")[-1]
                                            content_list.append(
                                                FastMCPImage(
                                                    data=response.content,
                                                    format=image_format,
                                                )
                                            )
                                    except Exception as e:
                                        logger.debug(f"Image download failed: {e}")
                            else:
                                # 假设为base64数据：直接使用
                                try:
                                    image_data = base64.b64decode(target_image)
                                    # 根据数据头判断格式
                                    if image_data.startswith(b"\xff\xd8\xff"):
                                        format_type = "jpeg"
                                    elif image_data.startswith(b"\x89PNG"):
                                        format_type = "png"
                                    else:
                                        format_type = "jpeg"  # 默认
                                    content_list.append(
                                        FastMCPImage(
                                            data=image_data, format=format_type
                                        )
                                    )
                                except Exception as e:
                                    logger.debug(f"Base64 decode failed: {e}")

                        return content_list
                    except Exception:
                        pass

                # 标准文本返回（包含图片URL引用）
                return format_structure_output(result, output_mode == "detailed")

            except Exception as e:
                if ctx:
                    await ctx.error(f"Structure analysis failed: {str(e)}")
                error_msg = f"Structure analysis failed: {str(e)}"
                return (
                    error_msg
                    if output_mode == "simple"
                    else json.dumps({"error": error_msg}, ensure_ascii=False)
                )
