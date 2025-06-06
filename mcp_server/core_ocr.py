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
from typing import Optional, Union, Dict, Any

# 第三方库导入
import numpy as np
from PIL import Image
from fastmcp import Context

# 配置logger - 仅错误级别
logger = logging.getLogger(__name__)

# 本地OCR库（可选导入）
try:
    from paddleocr import PaddleOCR, PPStructureV3

    LOCAL_OCR_AVAILABLE = True
except ImportError:
    LOCAL_OCR_AVAILABLE = False

# ==================== 统一引擎管理容器 ====================


class EngineContainer:
    """智能引擎容器 - 统一管理所有OCR产线"""

    def __init__(self):
        self._engines = {}
        self._api_config = None

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

        # PaddleOCR日志默认输出到stderr，无需额外抑制stdout
        if engine_name == "ocr":
            return PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_type="min",
                text_det_limit_side_len=736,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
            )
        elif engine_name == "structure":
            return PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_chart_recognition=False,
                use_seal_recognition=False,
                use_table_recognition=False,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                layout_detection_model_name="PP-DocLayout-M",
                device="cpu",
            )
        else:
            raise ValueError(f"Unknown engine: {engine_name}")

    def warmup_engines(self, ocr_source_types: list = None):
        """预热引擎 - eager初始化，避免首次运行延迟"""
        if not LOCAL_OCR_AVAILABLE:
            return  # API模式下无需预热

        if ocr_source_types is None:
            ocr_source_types = ["ocr", "structure"]  # 默认预热所有引擎

        for ocr_source_type in ocr_source_types:
            try:
                self.get_engine(ocr_source_type)
                logger.info(f"Engine '{ocr_source_type}' warmed up successfully")
            except Exception as e:
                logger.warning(f"Failed to warm up engine '{ocr_source_type}': {e}")


# 全局引擎容器实例
engines = EngineContainer()

# ==================== 兼容性接口（保持API不变） ====================


def configure_api(
    api_url: str,
    api_token: str = None,
    timeout: int = 30,
    service_type: str = "aistudio",
):
    """配置API连接参数 - 兼容接口"""
    engines.configure_api(api_url, service_type, api_token, timeout)


def is_api_mode() -> bool:
    """判断是否为API模式 - 兼容接口"""
    return engines.is_api_mode()


# ==================== 输入处理（简化版） ====================


def _validate_input_data(input_data: str) -> str:
    """验证输入数据的有效性 - 支持file path, URL, base64三种核心输入"""
    if not input_data or not input_data.strip():
        raise ValueError("Input data cannot be empty")

    input_data = input_data.strip()

    # 检查核心支持格式
    if (
        input_data.startswith(("http://", "https://", "file://", "data:"))
        or _is_file_path(input_data)
        or _is_base64_like(input_data)
    ):
        return input_data

    # 其他情况尝试作为文件路径处理
    return input_data


def process_input(input_data: str) -> Union[str, np.ndarray]:
    """统一智能输入处理 - 简化同步异步逻辑"""
    # 输入验证
    validated_input = _validate_input_data(input_data)

    if is_api_mode():
        # API模式：直接返回，让API调用函数处理转换
        return validated_input
    else:
        # 本地模式：优先文件路径，Base64转numpy
        if _is_file_path(validated_input):
            return validated_input
        elif validated_input.startswith(("http://", "https://")):
            return validated_input
        elif _is_base64_like(validated_input):
            return _base64_to_numpy(validated_input)
        return validated_input


def _base64_to_numpy(data: str) -> np.ndarray:
    """Base64转numpy数组"""
    if data.startswith("data:"):
        base64_data = data.split(",", 1)[1]
    else:
        base64_data = data

    image_bytes = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)


def _is_file_path(path: str) -> bool:
    """跨平台文件路径判断"""
    try:
        if path.startswith(("http://", "https://", "data:")):
            return False

        path_obj = Path(path)
        return (
            path_obj.is_absolute()
            or os.sep in path
            or "/" in path
            or path.startswith(("./", "../", ".\\", "..\\", "~"))
        )

    except (ValueError, OSError):
        return False


def _is_base64_like(s: str) -> bool:
    """Base64格式检查"""
    return len(s) > 10 and len(s) % 4 == 0 and re.match(r"^[A-Za-z0-9+/]*={0,2}$", s)


def _detect_file_type(input_data: str) -> int:
    """检测文件类型 (0=PDF, 1=Image)"""
    lower_data = input_data.lower()

    if "application/pdf" in lower_data or lower_data.endswith(".pdf"):
        return 0

    if _is_base64_like(input_data) and input_data.startswith("JVBERi"):
        return 0

    return 1


# ==================== API调用（使用Context简化） ====================


async def _call_aistudio_api(input_data: str, ctx: Context, **options) -> dict:
    """调用星河API"""
    if not engines._api_config:
        raise ValueError("API not configured")

    # 转换为Base64
    if _is_file_path(input_data):
        with open(input_data, "rb") as f:
            file_data = base64.b64encode(f.read()).decode("ascii")
    elif input_data.startswith("data:"):
        file_data = input_data.split(",", 1)[1] if "," in input_data else input_data
    elif input_data.startswith(("http://", "https://")):
        response = await ctx.http_request(method="GET", url=input_data)
        if response.status_code != 200:
            raise Exception(f"Failed to download URL: {response.status_code}")
        file_data = base64.b64encode(response.content).decode("ascii")
    else:
        file_data = input_data

    # 构建payload
    payload = {"file": file_data, "fileType": _detect_file_type(input_data)}

    if options.get("useDocOrientationClassify") is not None:
        payload["useDocOrientationClassify"] = options["useDocOrientationClassify"]
    if options.get("useDocUnwarping") is not None:
        payload["useDocUnwarping"] = options["useDocUnwarping"]

    # 构建headers
    headers = {"Content-Type": "application/json"}
    if engines._api_config["token"]:
        headers["Authorization"] = f'token {engines._api_config["token"]}'

    # 发送请求
    response = await ctx.http_request(
        method="POST",
        url=engines._api_config["url"],
        headers=headers,
        json=payload,
        timeout=engines._api_config.get("timeout", 30),
    )

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")

    response_json = response.json()
    error_code = response_json.get("errorCode", 0)

    if error_code != 0:
        error_msg = response_json.get("errorMsg", "Unknown API error")
        raise Exception(f"API failed (errorCode: {error_code}): {error_msg}")

    return response_json["result"]


# ==================== 结果解析（简化版） ====================


def _parse_ocr_result(raw_result) -> dict:
    """解析OCR结果 - 统一格式"""
    if is_api_mode():
        return _parse_api_ocr(raw_result)
    else:
        return _parse_local_ocr(raw_result)


def _build_ocr_result(
    rec_texts: list,
    rec_scores: list,
    rec_boxes_list: list,
    text_type: str,
    det_params: Any,
) -> dict:
    """通用OCR结果构建器 - 复用核心解析逻辑"""
    if not rec_texts:
        return {
            "text": "",
            "confidence": 0,
            "blocks": [],
            "text_type": text_type,
            "det_params": det_params,
        }

    texts, confidences, blocks = [], [], []

    for i, text in enumerate(rec_texts):
        if text and text.strip():
            conf = rec_scores[i] if i < len(rec_scores) else 0
            texts.append(text.strip())
            confidences.append(conf)

            block = {"text": text.strip(), "confidence": round(conf, 3)}

            # 添加边界框信息（如果有）- 支持多种边界框来源
            for bbox_source in rec_boxes_list:
                if i < len(bbox_source) and bbox_source[i] is not None:
                    # 保留原始边界框格式，不做标准化处理
                    block["bbox"] = bbox_source[i]
                    break

            blocks.append(block)

    return {
        "text": "\n".join(texts),
        "confidence": sum(confidences) / len(confidences) if confidences else 0,
        "blocks": blocks,
        "text_type": text_type,
        "det_params": det_params,
    }


def _parse_api_ocr(api_result: dict) -> dict:
    """解析星河API OCR结果 - 基于确定的API响应结构"""
    # ocrResults字段必定存在，直接访问
    ocr_results = api_result["ocrResults"]

    if not ocr_results:
        return _build_ocr_result([], [], [], "api", None)

    # 处理多个OCR结果（通常只有一个）
    all_texts, all_scores, all_boxes = [], [], []
    det_params = None

    for ocr_result in ocr_results:
        # prunedResult字段必定存在，直接访问
        pruned = ocr_result["prunedResult"]

        if isinstance(pruned, dict):
            # rec_texts等字段在API响应中是确定存在的
            rec_texts = pruned.get("rec_texts", [])
            rec_scores = pruned.get("rec_scores", [])
            rec_boxes = pruned.get("rec_boxes", [])

            all_texts.extend(rec_texts)
            all_scores.extend(rec_scores)
            all_boxes.extend(rec_boxes)

            # 获取检测参数（只取第一个）
            if det_params is None:
                det_params = pruned.get("text_det_params")

        elif isinstance(pruned, str) and pruned.strip():
            # 处理纯字符串结果
            all_texts.append(pruned.strip())
            all_scores.append(1.0)  # 默认置信度
            all_boxes.append(None)

    # 使用通用构建器
    return _build_ocr_result(all_texts, all_scores, [all_boxes], "api", det_params)


def _parse_local_ocr(raw_result) -> dict:
    """解析本地OCR结果"""
    if not raw_result or not raw_result[0]:
        return _build_ocr_result([], [], [], "unknown", None)

    ocr_result = raw_result[0]

    # v5格式处理 - 直接访问字典键，因为ocr_result已确认为字典类型
    rec_texts = ocr_result.get("rec_texts", [])
    rec_scores = ocr_result.get("rec_scores", [])
    rec_boxes = ocr_result.get("rec_boxes", [])
    rec_polys = ocr_result.get("rec_polys", [])

    # 提取元数据
    text_type = ocr_result.get("text_type", "general")
    det_params = ocr_result.get("text_det_params", {})

    # 使用通用构建器 - 支持多种边界框来源（boxes优先，polys备选）
    return _build_ocr_result(
        rec_texts, rec_scores, [rec_boxes, rec_polys], text_type, det_params
    )


def _parse_structure_result(raw_result) -> dict:
    """解析结构分析结果 - 统一格式"""
    if is_api_mode():
        return _parse_api_structure(raw_result)
    else:
        return _parse_local_structure(raw_result)


def _parse_api_structure(api_result: dict) -> dict:
    """解析星河API结构结果 - 基于确定的API响应结构"""
    # layoutParsingResults字段必定存在，直接访问
    layout_results = api_result["layoutParsingResults"]

    if not layout_results:
        return {"markdown": "", "pages": [], "has_images": False}

    markdown_parts = []
    pages = []
    has_images = False

    for i, res in enumerate(layout_results):
        # markdown字段在API响应中是确定存在的
        markdown_data = res.get("markdown", {})  # 这里保留get，因为可能确实为空
        if markdown_data.get("text"):
            text = markdown_data["text"]
            markdown_parts.append(text)
            pages.append(
                {
                    "page": i,
                    "content": text,
                    "has_images": bool(markdown_data.get("images")),
                }
            )
            if markdown_data.get("images"):
                has_images = True

    return {
        "markdown": "\n".join(markdown_parts),
        "pages": pages,
        "has_images": has_images,
    }


def _parse_local_structure(raw_results) -> dict:
    """解析本地结构结果"""
    if not raw_results:
        return {"markdown": "", "pages": [], "has_images": False}

    markdown_parts = []
    pages = []

    for i, result in enumerate(raw_results):
        text = ""
        if hasattr(result, "markdown") and result.markdown:
            if isinstance(result.markdown, dict):
                text = result.markdown.get("text", str(result.markdown))
            else:
                text = str(result.markdown)

        if text:
            markdown_parts.append(text)
            pages.append(
                {"page": i, "content": text, "has_images": False}  # 本地模式简化处理
            )

    return {"markdown": "\n".join(markdown_parts), "pages": pages, "has_images": False}


# ==================== 格式化输出（简化版） ====================


def format_ocr_output(result: dict, detailed: bool = False) -> str:
    """格式化OCR输出 - 统一L1/L2格式"""
    if not result["text"].strip():
        error_msg = "❌ No text detected"
        return (
            error_msg
            if not detailed
            else json.dumps(
                {
                    "error": "No text detected",
                    "text": "",
                    "blocks": [],
                    "meta": {
                        "block_count": 0,
                        "avg_confidence": 0,
                        "text_type": "unknown",
                    },
                },
                ensure_ascii=False,
            )
        )

    if detailed:
        # L2: 完整结构化输出
        output = {
            "text": result["text"],
            "blocks": result["blocks"],
            "meta": {
                "block_count": len(result["blocks"]),
                "avg_confidence": (
                    round(result["confidence"], 3) if result["confidence"] > 0 else None
                ),
                "has_coordinates": any("bbox" in block for block in result["blocks"]),
                "text_type": result.get("text_type", "unknown"),
                "source_type": "api" if is_api_mode() else "local",
            },
        }

        # 添加检测参数（如果有且有用）
        if result.get("det_params") and isinstance(result["det_params"], dict):
            # 只包含关键参数，避免信息过载
            key_params = {
                k: v
                for k, v in result["det_params"].items()
                if k in ["box_thresh", "limit_side_len", "limit_type", "thresh"]
            }
            if key_params:
                output["meta"]["detection_params"] = key_params

        return json.dumps(output, ensure_ascii=False, indent=2)
    else:
        # L1: 简洁文本输出 + 关键信息
        output = result["text"]

        # 添加简洁的元信息
        info_parts = []
        if result["confidence"] > 0:
            info_parts.append(f"置信度: {(result['confidence'] * 100):.1f}%")
        info_parts.append(f"{len(result['blocks'])}个文本块")
        if result.get("text_type") and result["text_type"] != "unknown":
            info_parts.append(f"类型: {result['text_type']}")

        if info_parts:
            output += f"\n\n📊 {' | '.join(info_parts)}"

        return output


def format_structure_output(result: dict, detailed: bool = False) -> str:
    """格式化结构分析输出 - 统一L1/L2"""
    if not result["markdown"].strip():
        error_msg = "❌ No document structure detected"
        return (
            error_msg
            if not detailed
            else json.dumps(
                {
                    "error": "No structure detected",
                    "markdown": "",
                    "pages": [],
                    "meta": {"page_count": 0, "has_images": False},
                },
                ensure_ascii=False,
            )
        )

    if detailed:
        # L2: 完整结构化输出
        output = {
            "markdown": result["markdown"],
            "pages": result["pages"],
            "meta": {
                "page_count": len(result["pages"]),
                "has_images": result["has_images"],
                "source_type": "api" if is_api_mode() else "local",
            },
        }
        return json.dumps(output, ensure_ascii=False, indent=2)
    else:
        # L1: 纯markdown输出
        return result["markdown"]


# ==================== MCP工具注册（简化版） ====================


def register_tools(mcp, ocr_source_type: str = "local", **api_config):
    """注册MCP工具 - 支持本地、AI Studio、用户服务三种模式"""

    # 配置API（如果是API模式）
    if ocr_source_type in ["aistudio", "user_service"]:
        # 明确传递服务类型，而不是通过URL推测
        engines.configure_api(service_type=ocr_source_type, **api_config)
    else:
        # 本地模式：预热引擎避免首次运行延迟
        engines.warmup_engines()

    # 智能描述引擎类型
    if engines.is_api_mode():
        if engines._api_config["is_user_service"]:
            source_desc = "用户服务API"
        else:
            source_desc = "星河API"
    else:
        source_desc = "本地PaddleOCR"

    @mcp.tool()
    async def ocr_text(
        input_data: str,
        output_mode: str = "simple",
        file_type: str = "auto",
        useDocOrientationClassify: bool = True,
        useDocUnwarping: bool = True,
        ctx: Optional[Context] = None,
    ) -> str:
        f"""Extract text from images and PDFs using {source_desc}.

        Args:
            input_data: 文件路径、URL或Base64数据
            output_mode: "simple" (L1简洁) 或 "detailed" (L2详细)
            file_type: 文件类型 ("auto", "pdf", "image") - API模式
            useDocOrientationClassify: 文档方向分类 - API模式
            useDocUnwarping: 文档图像校正 - API模式
        """
        try:
            # 准备API参数
            api_kwargs = {
                "file_type": file_type,
                "useDocOrientationClassify": useDocOrientationClassify,
                "useDocUnwarping": useDocUnwarping,
            }

            # 输入处理
            processed_input = process_input(input_data)

            if engines.is_api_mode():
                # API模式 (AI Studio 或用户服务)
                if not ctx:
                    raise ValueError("Context required for API mode")
                raw_result = await _call_aistudio_api(
                    processed_input, ctx, **api_kwargs
                )
            else:
                # 本地OCR模式 - 使用引擎容器，异步执行避免阻塞event loop
                ocr = engines.get_engine("ocr")
                # 使用run_in_executor在线程池中执行耗时的同步OCR操作
                loop = asyncio.get_running_loop()
                raw_result = await loop.run_in_executor(
                    None, ocr.predict, processed_input
                )

            result = _parse_ocr_result(raw_result)
            return format_ocr_output(result, output_mode == "detailed")

        except Exception as e:
            error_msg = f"OCR failed: {str(e)}"
            return (
                error_msg
                if output_mode == "simple"
                else json.dumps({"error": error_msg}, ensure_ascii=False)
            )

    @mcp.tool()
    async def analyze_structure(
        input_data: str,
        output_mode: str = "detailed",
        file_type: str = "auto",
        ctx: Optional[Context] = None,
    ) -> str:
        f"""Analyze document structure using {source_desc}.

        Args:
            input_data: 文件路径、URL或Base64数据
            output_mode: "simple" (L1纯markdown) 或 "detailed" (L2结构化)
            file_type: 文件类型 ("auto", "pdf", "image") - API模式
        """
        try:
            # 准备API参数
            api_kwargs = {"file_type": file_type}

            # 输入处理
            processed_input = process_input(input_data)

            if engines.is_api_mode():
                # API模式 (AI Studio 或用户服务)
                if not ctx:
                    raise ValueError("Context required for API mode")
                raw_result = await _call_aistudio_api(
                    processed_input, ctx, **api_kwargs
                )
            else:
                # 本地结构分析模式 - 使用引擎容器，异步执行避免阻塞event loop
                structure = engines.get_engine("structure")
                # 使用run_in_executor在线程池中执行耗时的同步结构分析操作
                loop = asyncio.get_running_loop()
                raw_result = await loop.run_in_executor(
                    None, structure.predict, processed_input
                )

            result = _parse_structure_result(raw_result)
            return format_structure_output(result, output_mode == "detailed")

        except Exception as e:
            error_msg = f"Structure analysis failed: {str(e)}"
            return (
                error_msg
                if output_mode == "simple"
                else json.dumps({"error": error_msg}, ensure_ascii=False)
            )
