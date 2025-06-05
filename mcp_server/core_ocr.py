"""
PaddleOCR MCP Server
采用FastMCP v2
"""

import base64
import io
import json
import os
import re
import sys
from contextlib import contextmanager
from io import StringIO
from typing import Optional, Union, Dict, Any

import numpy as np
from PIL import Image
from fastmcp import Context

# 本地OCR库（可选导入）
try:
    from paddleocr import PaddleOCR, PPStructureV3
    LOCAL_OCR_AVAILABLE = True
except ImportError:
    LOCAL_OCR_AVAILABLE = False

# 全局引擎实例（延迟初始化）
_ocr_engine = None
_structure_engine = None
_api_config = None


# ==================== 配置管理 ====================

def configure_api(api_url: str, api_token: str = None, timeout: int = 30):
    """配置API参数 - 支持AI Studio和本地服务"""
    global _api_config
    
    # 智能检测服务类型
    is_local_service = any(indicator in api_url.lower() for indicator in [
        'localhost', '127.0.0.1', '10.', '192.168.', '172.'
    ])
    
    _api_config = {
        'url': api_url,
        'token': api_token,
        'timeout': timeout,
        'is_local_service': is_local_service,
        'service_type': 'layout-parsing' if 'layout-parsing' in api_url.lower() else 'ocr'
    }


def is_api_mode() -> bool:
    """判断是否为API模式"""
    return _api_config is not None


# ==================== 输入处理（简化版） ====================

def process_input(input_path: str) -> Union[str, np.ndarray]:
    """智能输入处理 - 根据模式自动选择最佳方式"""
    
    if is_api_mode():
        # API模式：转换为Base64
        return _to_base64(input_path)
    else:
        # 本地模式：优先文件路径
        if _is_file_path(input_path):
            return input_path
        elif input_path.startswith(('http://', 'https://')):
            return input_path
        elif _is_base64_like(input_path):
            return _base64_to_numpy(input_path)
        return input_path


async def process_input_async(input_path: str, ctx: Optional[Context] = None) -> Union[str, np.ndarray]:
    """异步智能输入处理 - 支持URL下载"""
    
    if is_api_mode():
        # API模式：转换为Base64
        return await _to_base64_async(input_path, ctx)
    else:
        # 本地模式：优先文件路径
        if _is_file_path(input_path):
            return input_path
        elif input_path.startswith(('http://', 'https://')):
            return input_path
        elif _is_base64_like(input_path):
            return _base64_to_numpy(input_path)
        return input_path


def _to_base64(input_path: str) -> str:
    """转换任何输入为Base64（同步版本）"""
    # Data URI
    if input_path.startswith('data:'):
        return input_path.split(',', 1)[1] if ',' in input_path else input_path
    
    # 已经是Base64
    if _is_base64_like(input_path) and not _is_file_path(input_path):
        return input_path
    
    # 文件路径
    if _is_file_path(input_path):
        with open(input_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    # URL需要异步处理，这里抛出异常提示
    if input_path.startswith(('http://', 'https://')):
        raise ValueError("URL input requires async processing. Use process_input_async instead.")
    
    # 默认当作Base64
    return input_path


async def _to_base64_async(input_path: str, ctx: Optional[Context] = None) -> str:
    """转换任何输入为Base64（异步版本，支持URL下载）"""
    # Data URI
    if input_path.startswith('data:'):
        return input_path.split(',', 1)[1] if ',' in input_path else input_path
    
    # 已经是Base64
    if _is_base64_like(input_path) and not _is_file_path(input_path):
        return input_path
    
    # URL处理 - 使用Context下载
    if input_path.startswith(('http://', 'https://')):
        if not ctx:
            raise ValueError("Context required for URL download")
        
        response = await ctx.http_request(method="GET", url=input_path, timeout=15)
        if response.status_code != 200:
            raise Exception(f"Failed to download URL: {response.status_code}")
        
        return base64.b64encode(response.content).decode('utf-8')
    
    # 文件路径
    if _is_file_path(input_path):
        with open(input_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    # 默认当作Base64
    return input_path


def _base64_to_numpy(data: str) -> np.ndarray:
    """Base64转numpy数组（本地OCR使用）"""
    if data.startswith('data:'):
        base64_data = data.split(',', 1)[1]
    else:
        base64_data = data
    
    image_bytes = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # 转RGB + 限制大小
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            bg = Image.new('RGB', image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[-1])
            image = bg
        else:
            image = image.convert('RGB')
    
    # 限制图像大小
    if image.width * image.height > 4096 * 4096:
        scale = (4096 * 4096 / (image.width * image.height)) ** 0.5
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)
    
    return np.array(image)


def _is_file_path(path: str) -> bool:
    """简单的文件路径判断"""
    return (path.startswith(('/', './', '../', '~')) or 
            (len(path) > 3 and path[1] == ':'))


def _is_base64_like(s: str) -> bool:
    """简单的Base64格式检查"""
    return len(s) > 10 and len(s) % 4 == 0 and re.match(r'^[A-Za-z0-9+/]*={0,2}$', s)


def _detect_file_type(input_path: str) -> int:
    """检测文件类型 (0=PDF, 1=Image)"""
    lower_path = input_path.lower()
    
    if 'application/pdf' in lower_path or lower_path.endswith('.pdf'):
        return 0
    
    # Base64 PDF检查
    if _is_base64_like(input_path) and input_path.startswith('JVBERi'):
        return 0
    
    return 1  # 默认图像


# ==================== 本地OCR引擎（简化版） ====================

@contextmanager
def _suppress_output():
    """抑制PaddleOCR输出"""
    original = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = original


def _get_ocr_engine():
    """获取OCR引擎实例"""
    global _ocr_engine
    if _ocr_engine is None:
        if not LOCAL_OCR_AVAILABLE:
            raise RuntimeError("PaddleOCR not available. Please install paddleocr.")
        with _suppress_output():
            _ocr_engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_type="min",
                text_det_limit_side_len=736,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec"
            )
    return _ocr_engine


def _get_structure_engine():
    """获取结构分析引擎实例"""
    global _structure_engine
    if _structure_engine is None:
        if not LOCAL_OCR_AVAILABLE:
            raise RuntimeError("PaddleOCR not available. Please install paddleocr.")
        with _suppress_output():
            _structure_engine = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_chart_recognition=False,
                use_seal_recognition=False,  
                use_table_recognition=False,    
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                layout_detection_model_name="PP-DocLayout-M",  
                device="cpu"
            )
    return _structure_engine


# ==================== API调用（使用Context简化） ====================

async def _call_aistudio_api(input_data: str, ctx: Context, **options) -> dict:
    """调用API - 统一支持AI Studio和本地服务"""
    if not _api_config:
        raise ValueError("API not configured")
    
    # 构建请求payload
    payload = {
        'file': input_data,
        'fileType': _detect_file_type(input_data)
    }
    
    # OCR服务的额外参数
    if _api_config['service_type'] == 'ocr':
        payload.update({
            'useDocOrientationClassify': options.get('useDocOrientationClassify', False),
            'useDocUnwarping': options.get('useDocUnwarping', False),
            'useTextlineOrientation': options.get('useTextlineOrientation', False)
        })
    
    # 智能构建请求头 - 本地服务不需要token
    headers = {'Content-Type': 'application/json'}
    if not _api_config['is_local_service'] and _api_config['token']:
        headers['Authorization'] = f'token {_api_config["token"]}'
    
    # 使用Context发送HTTP请求
    response = await ctx.http_request(
        method="POST",
        url=_api_config['url'],
        headers=headers,
        json=payload,
        timeout=_api_config['timeout']
    )
    
    if response.status_code != 200:
        service_type = "本地服务" if _api_config['is_local_service'] else "AI Studio"
        raise Exception(f"{service_type} API error {response.status_code}: {response.text}")
    
    return response.json()['result']


# ==================== 结果解析（简化版） ====================

def _parse_ocr_result(raw_result) -> dict:
    """解析OCR结果 - 统一格式"""
    if is_api_mode():
        return _parse_api_ocr(raw_result)
    else:
        return _parse_local_ocr(raw_result)


def _parse_api_ocr(api_result: dict) -> dict:
    """解析星河API OCR结果"""
    if not api_result.get('ocrResults'):
        return {"text": "", "confidence": 0, "blocks": [], "text_type": "api", "det_params": None}
    
    texts, confidences, blocks = [], [], []
    
    for ocr_result in api_result['ocrResults']:
        pruned = ocr_result.get('prunedResult', {})
        
        if isinstance(pruned, dict):
            for i, text in enumerate(pruned.get('rec_texts', [])):
                if text and text.strip():
                    conf = pruned.get('rec_scores', [])[i] if i < len(pruned.get('rec_scores', [])) else 0
                    texts.append(text.strip())
                    confidences.append(conf)
                    
                    block = {"text": text.strip(), "confidence": round(conf, 3)}
                    
                    # 添加边界框信息（如果有）
                    rec_boxes = pruned.get('rec_boxes', [])
                    if i < len(rec_boxes) and rec_boxes[i] is not None:
                        bbox = _normalize_bbox(rec_boxes[i])
                        if bbox:
                            block["bbox"] = bbox
                    
                    blocks.append(block)
        elif isinstance(pruned, str) and pruned.strip():
            texts.append(pruned.strip())
            blocks.append({"text": pruned.strip(), "confidence": None})
    
    return {
        "text": "\n".join(texts),
        "confidence": sum(confidences) / len(confidences) if confidences else 0,
        "blocks": blocks,
        "text_type": "api", 
        "det_params": api_result.get('ocrResults', [{}])[0].get('prunedResult', {}).get('text_det_params')
    }


def _parse_local_ocr(raw_result) -> dict:
    """解析本地OCR结果"""
    if not raw_result or not raw_result[0]:
        return {"text": "", "confidence": 0, "blocks": [], "text_type": "unknown", "det_params": None}
    
    ocr_result = raw_result[0]
    
    # v5格式处理
    def safe_get(obj, key, default=None):
        if hasattr(obj, key):
            return getattr(obj, key)
        elif isinstance(obj, dict):
            return obj.get(key, default)
        return default
    
    rec_texts = safe_get(ocr_result, 'rec_texts', [])
    rec_scores = safe_get(ocr_result, 'rec_scores', [])
    rec_boxes = safe_get(ocr_result, 'rec_boxes', [])
    rec_polys = safe_get(ocr_result, 'rec_polys', [])
    
    # 提取元数据
    text_type = safe_get(ocr_result, 'text_type', 'general')
    det_params = safe_get(ocr_result, 'text_det_params', {})
    
    texts, confidences, blocks = [], [], []
    
    for i, text in enumerate(rec_texts):
        if text and text.strip():
            conf = rec_scores[i] if i < len(rec_scores) else 0
            texts.append(text.strip())
            confidences.append(conf)
            
            block = {"text": text.strip(), "confidence": round(conf, 3)}
            
            # 统一处理边界框 - 优先使用rec_boxes
            bbox = None
            if i < len(rec_boxes) and rec_boxes[i] is not None:
                bbox = _normalize_bbox(rec_boxes[i])
            elif i < len(rec_polys) and rec_polys[i] is not None:
                bbox = _normalize_bbox(rec_polys[i])
            
            if bbox:
                block["bbox"] = bbox
            
            blocks.append(block)
    
    # 如果上述v5格式解析失败，尝试遗留格式
    if not texts and isinstance(raw_result[0], list):
        legacy_result = _parse_legacy_ocr_format(raw_result[0])
        legacy_result.update({"text_type": "legacy", "det_params": None})
        return legacy_result
    
    return {
        "text": "\n".join(texts),
        "confidence": sum(confidences) / len(confidences) if confidences else 0,
        "blocks": blocks,
        "text_type": text_type,
        "det_params": det_params
    }


def _parse_legacy_ocr_format(text_lines) -> dict:
    """解析遗留OCR格式（旧版本兼容）"""
    texts, confidences, blocks = [], [], []
    
    for line_result in text_lines:
        if isinstance(line_result, list) and len(line_result) >= 2:
            bbox_data = line_result[0]  # 边界框
            text_data = line_result[1]  # 文本和置信度
            
            if isinstance(text_data, tuple) and len(text_data) >= 2:
                text, confidence = text_data[0], text_data[1]
                if text and text.strip():
                    texts.append(text.strip())
                    confidences.append(confidence)
                    
                    block = {
                        "text": text.strip(),
                        "confidence": round(confidence, 3) if confidence > 0 else None
                    }
                    
                    # 添加边界框（遗留格式）
                    if bbox_data and len(bbox_data) >= 4:
                        bbox = _normalize_bbox(bbox_data)
                        if bbox:
                            block["bbox"] = bbox
                    
                    blocks.append(block)
        
    return {
        "text": "\n".join(texts),
        "confidence": sum(confidences) / len(confidences) if confidences else 0,
        "blocks": blocks
    }


def _normalize_bbox(bbox) -> list:
    """标准化边界框格式 - 转换为 [x1, y1, x2, y2]"""
    try:
        # 处理numpy数组
        if hasattr(bbox, 'tolist'):
            bbox = bbox.tolist()
        
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        
        # 如果已经是4个坐标格式 [x1,y1,x2,y2]
        if len(bbox) == 4 and all(isinstance(coord, (int, float)) for coord in bbox):
            return [round(float(coord), 1) for coord in bbox]
        
        # 如果是8个坐标的多边形格式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        elif len(bbox) == 4 and all(isinstance(point, (list, tuple)) and len(point) == 2 for point in bbox):
            x_coords = [float(point[0]) for point in bbox]
            y_coords = [float(point[1]) for point in bbox]
            return [
                round(min(x_coords), 1), round(min(y_coords), 1),
                round(max(x_coords), 1), round(max(y_coords), 1)
            ]
        
        # 如果是扁平化的8个坐标 [x1,y1,x2,y2,x3,y3,x4,y4]
        elif len(bbox) == 8:
            x_coords = [float(bbox[i]) for i in range(0, 8, 2)]
            y_coords = [float(bbox[i]) for i in range(1, 8, 2)]
            return [
                round(min(x_coords), 1), round(min(y_coords), 1),
                round(max(x_coords), 1), round(max(y_coords), 1)
            ]
        
        # 尝试处理嵌套数组格式（numpy特有）
        elif len(bbox) > 4 and hasattr(bbox[0], '__iter__'):
            # 展平所有坐标点
            all_points = []
            for item in bbox:
                if hasattr(item, 'tolist'):
                    item = item.tolist()
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    all_points.append([float(item[0]), float(item[1])])
            
            if all_points:
                x_coords = [point[0] for point in all_points]
                y_coords = [point[1] for point in all_points]
                return [
                    round(min(x_coords), 1), round(min(y_coords), 1),
                    round(max(x_coords), 1), round(max(y_coords), 1)
                ]
        
        return None
    except (ValueError, TypeError, IndexError):
        return None


def _parse_structure_result(raw_result) -> dict:
    """解析结构分析结果 - 统一格式"""
    if is_api_mode():
        return _parse_api_structure(raw_result)
    else:
        return _parse_local_structure(raw_result)


def _parse_api_structure(api_result: dict) -> dict:
    """解析星河API结构结果"""
    if not api_result.get('layoutParsingResults'):
        return {"markdown": "", "pages": [], "has_images": False}
    
    markdown_parts = []
    pages = []
    has_images = False
    
    for i, res in enumerate(api_result['layoutParsingResults']):
        markdown_data = res.get('markdown', {})
        if markdown_data.get('text'):
            text = markdown_data['text']
            markdown_parts.append(text)
            pages.append({
                "page": i,
                "content": text,
                "has_images": bool(markdown_data.get('images'))
            })
            if markdown_data.get('images'):
                has_images = True
    
    return {
        "markdown": "\n".join(markdown_parts),
        "pages": pages,
        "has_images": has_images
    }


def _parse_local_structure(raw_results) -> dict:
    """解析本地结构结果"""
    if not raw_results:
        return {"markdown": "", "pages": [], "has_images": False}
    
    markdown_parts = []
    pages = []
    
    for i, result in enumerate(raw_results):
        text = ""
        if hasattr(result, 'markdown') and result.markdown:
            if isinstance(result.markdown, dict):
                text = result.markdown.get('text', str(result.markdown))
            else:
                text = str(result.markdown)
        
        if text:
            markdown_parts.append(text)
            pages.append({
                "page": i,
                "content": text,
                "has_images": False  # 本地模式简化处理
            })
    
    return {
        "markdown": "\n".join(markdown_parts),
        "pages": pages,
        "has_images": False
    }


# ==================== 格式化输出（简化版） ====================

def format_ocr_output(result: dict, detailed: bool = False) -> str:
    """格式化OCR输出 - 统一L1/L2格式"""
    if not result["text"].strip():
        error_msg = "❌ No text detected"
        return error_msg if not detailed else json.dumps({
            "error": "No text detected", 
            "text": "", 
            "blocks": [],
            "meta": {"block_count": 0, "avg_confidence": 0, "text_type": "unknown"}
        }, ensure_ascii=False)
    
    if detailed:
        # L2: 完整结构化输出
        output = {
            "text": result["text"],
            "blocks": result["blocks"],
            "meta": {
                "block_count": len(result["blocks"]),
                "avg_confidence": round(result["confidence"], 3) if result["confidence"] > 0 else None,
                "has_coordinates": any("bbox" in block for block in result["blocks"]),
                "text_type": result.get("text_type", "unknown"),
                "engine_type": "api" if is_api_mode() else "local"
            }
        }
        
        # 添加检测参数（如果有且有用）
        if result.get("det_params") and isinstance(result["det_params"], dict):
            # 只包含关键参数，避免信息过载
            key_params = {k: v for k, v in result["det_params"].items() 
                         if k in ['box_thresh', 'limit_side_len', 'limit_type', 'thresh']}
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
        return error_msg if not detailed else json.dumps({
            "error": "No structure detected", 
            "markdown": "", 
            "pages": [],
            "meta": {"page_count": 0, "has_images": False}
        }, ensure_ascii=False)
    
    if detailed:
        # L2: 完整结构化输出
        output = {
            "markdown": result["markdown"],
            "pages": result["pages"],
            "meta": {
                "page_count": len(result["pages"]),
                "has_images": result["has_images"],
                "engine_type": "api" if is_api_mode() else "local"
            }
        }
        return json.dumps(output, ensure_ascii=False, indent=2)
    else:
        # L1: 纯markdown输出
        return result["markdown"]


# ==================== MCP工具注册（简化版） ====================

def register_tools(mcp, engine_type: str = "local", **api_config):
    """注册MCP工具 - 支持本地、AI Studio、本地服务三种模式"""
    
    # 配置API（如果是API模式）
    if engine_type in ["aistudio", "local_service"]:
        configure_api(**api_config)
    
    # 智能描述引擎类型
    if is_api_mode():
        if _api_config['is_local_service']:
            engine_desc = "本地服务API"
        else:
            engine_desc = "星河API"
    else:
        engine_desc = "本地PaddleOCR"
    
    @mcp.tool()
    async def ocr_text(
        input_path: str, 
        output_mode: str = "simple",
        file_type: str = "auto",
        useDocOrientationClassify: bool = True,
        useDocUnwarping: bool = True,
        ctx: Optional[Context] = None
    ) -> str:
        f"""Extract text from images and PDFs using {engine_desc}.
        
        Args:
            input_path: 文件路径、URL或Base64数据
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
                "useDocUnwarping": useDocUnwarping
            }
            
            # 使用异步输入处理（支持URL下载）
            if is_api_mode():
                processed_input = await process_input_async(input_path, ctx)
            else:
                processed_input = process_input(input_path)
            
            if is_api_mode():
                # API模式 (AI Studio 或本地服务)
                if not ctx:
                    raise ValueError("Context required for API mode")
                raw_result = await _call_aistudio_api(processed_input, ctx, **api_kwargs)
            else:
                # 本地OCR模式
                ocr = _get_ocr_engine()
                with _suppress_output():
                    raw_result = ocr.predict(processed_input)
            
            result = _parse_ocr_result(raw_result)
            return format_ocr_output(result, output_mode == "detailed")
            
        except Exception as e:
            error_msg = f"OCR failed: {str(e)}"
            return error_msg if output_mode == "simple" else json.dumps({"error": error_msg}, ensure_ascii=False)
    
    @mcp.tool()
    async def analyze_structure(
        input_path: str,
        output_mode: str = "detailed",
        file_type: str = "auto",
        ctx: Optional[Context] = None
    ) -> str:
        f"""Analyze document structure using {engine_desc}.
        
        Args:
            input_path: 文件路径、URL或Base64数据
            output_mode: "simple" (L1纯markdown) 或 "detailed" (L2结构化)
            file_type: 文件类型 ("auto", "pdf", "image") - API模式
        """
        try:
            # 准备API参数
            api_kwargs = {"file_type": file_type}
            
            # 使用异步输入处理（支持URL下载）
            if is_api_mode():
                processed_input = await process_input_async(input_path, ctx)
            else:
                processed_input = process_input(input_path)
            
            if is_api_mode():
                # API模式 (AI Studio 或本地服务)
                if not ctx:
                    raise ValueError("Context required for API mode")
                raw_result = await _call_aistudio_api(processed_input, ctx, **api_kwargs)
            else:
                # 本地结构分析模式
                structure = _get_structure_engine()
                with _suppress_output():
                    raw_result = structure.predict(processed_input)
            
            result = _parse_structure_result(raw_result)
            return format_structure_output(result, output_mode == "detailed")
            
        except Exception as e:
            error_msg = f"Structure analysis failed: {str(e)}"
            return error_msg if output_mode == "simple" else json.dumps({"error": error_msg}, ensure_ascii=False) 