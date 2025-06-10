# 标准库导入
import abc
import asyncio
import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 第三方库导入
import httpx
import numpy as np
import filetype
from PIL import Image
from fastmcp import Context, Image as FastMCPImage


# ==================== 基础抽象类 ====================


class PipelineHandler(abc.ABC):
    """
    统一产线处理器 - 模板方法模式

    模板: 定义所有产线的统一处理流程
    契约: 强制子类实现差异化的解析和格式化逻辑
    """

    def __init__(self, engines):
        self.engines = engines

    async def process(
        self, input_data: str, output_mode: str, ctx: Optional[Context] = None
    ) -> Union[str, List]:
        """
        [模板] 统一的产线处理流程

        包含95%的共同逻辑：
        - 日志记录
        - API/Local模式判断
        - 引擎调用
        - 错误处理
        - 结果格式化
        """
        try:
            # 1. 开始处理日志
            if ctx:
                pipeline_name = self.get_pipeline_name()
                source_desc = self._get_source_description()
                await ctx.info(
                    f"Starting {pipeline_name} processing using {source_desc}"
                )

            # 2. 根据模式调用不同的处理逻辑
            if self.engines.is_api_mode():
                if not ctx:
                    raise ValueError("Context required for API mode")
                raw_result = await self._call_api(input_data)
                result = self._parse_api_result(raw_result)
            else:
                # 本地模式：输入预处理
                processed_input = self._process_input_for_local(input_data)

                # 获取引擎并异步执行
                engine = self.engines.get_engine(self.get_engine_name())
                loop = asyncio.get_running_loop()
                raw_result = await loop.run_in_executor(
                    None, engine.predict, processed_input
                )

                result = self._parse_local_result(raw_result)

            # 3. 完成处理日志
            if ctx:
                await self._log_completion_stats(ctx, result)

            # 4. 格式化输出
            return self._format_output(result, output_mode == "detailed")

        except Exception as e:
            # 5. 统一错误处理
            if ctx:
                await ctx.error(f"{self.get_pipeline_name()} failed: {str(e)}")
            return self._handle_error(str(e), output_mode)

    # ==================== [契约] 子类必须实现的抽象方法 ====================

    @abc.abstractmethod
    def get_pipeline_name(self) -> str:
        """获取产线名称 (用于日志和错误信息)"""
        pass

    @abc.abstractmethod
    def get_engine_name(self) -> str:
        """获取引擎名称 (本地模式使用)"""
        pass

    @abc.abstractmethod
    def _parse_api_result(self, api_result: Dict) -> Dict:
        """解析API返回的原始结果为统一格式"""
        pass

    @abc.abstractmethod
    def _parse_local_result(self, local_result: Any) -> Dict:
        """解析本地引擎返回的原始结果为统一格式"""
        pass

    @abc.abstractmethod
    def _format_output(self, result: Dict, detailed: bool) -> Union[str, List]:
        """将统一结果格式化为L1(simple)或L2(detailed)输出"""
        pass

    @abc.abstractmethod
    async def _log_completion_stats(self, ctx: Context, result: Dict):
        """记录处理完成的统计信息"""
        pass

    # ==================== [模板] 共享的辅助方法 ====================

    def _get_source_description(self) -> str:
        """获取数据源描述"""
        if self.engines.is_api_mode():
            return (
                "用户服务API"
                if self.engines._api_config["is_user_service"]
                else "星河API"
            )
        else:
            return "本地PaddleOCR"

    async def _call_api(self, input_data: str) -> dict:
        """调用API - 统一的API调用逻辑"""
        if not self.engines._api_config:
            raise ValueError("API not configured")

        # 文件路径 -> base64，其他原样传递
        if self._is_file_path(input_data):
            with open(input_data, "rb") as f:
                file_data = base64.b64encode(f.read()).decode("ascii")
        else:
            file_data = input_data

        # API需要文件类型参数
        payload = {"file": file_data, "fileType": self._detect_file_type(input_data)}

        # 构建headers
        headers = {"Content-Type": "application/json"}
        if self.engines._api_config["token"]:
            headers["Authorization"] = f'token {self.engines._api_config["token"]}'

        # 发送请求
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.engines._api_config["url"],
                headers=headers,
                json=payload,
                timeout=self.engines._api_config.get("timeout", 30),
            )

        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")

        response_json = response.json()
        error_code = response_json.get("errorCode", 0)

        if error_code != 0:
            error_msg = response_json.get("errorMsg", "Unknown API error")
            raise RuntimeError(f"API failed (errorCode: {error_code}): {error_msg}")

        return response_json["result"]

    def _process_input_for_local(self, input_data: str):
        """本地模式输入处理：base64转numpy，文件路径直接用"""
        if self._is_base64(input_data):
            if input_data.startswith("data:"):
                base64_data = input_data.split(",", 1)[1]
            else:
                base64_data = input_data
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_bytes))
            return np.array(image)
        return input_data

    def _handle_error(self, error_msg: str, output_mode: str) -> str:
        """统一错误处理"""
        pipeline_error = f"{self.get_pipeline_name()} failed: {error_msg}"
        return (
            pipeline_error
            if output_mode == "simple"
            else json.dumps({"error": pipeline_error}, ensure_ascii=False)
        )

    # ==================== 输入处理辅助方法 ====================

    @staticmethod
    def _is_file_path(data: str) -> bool:
        """简单文件路径判断"""
        return data.startswith(("/", "./", "../")) or "\\" in data

    @staticmethod
    def _is_base64(data: str) -> bool:
        """简单base64判断"""
        return (
            len(data) > 100
            and data.replace("+", "").replace("/", "").replace("=", "").isalnum()
        )

    @staticmethod
    def _detect_file_type(input_data: str) -> int:
        """检测文件类型：0=PDF，1=图片"""
        # 文件路径：检查扩展名
        if PipelineHandler._is_file_path(input_data):
            return 0 if input_data.lower().endswith(".pdf") else 1

        # Base64：检查PDF魔术字节
        if PipelineHandler._is_base64(input_data) and input_data.startswith("JVBERi"):
            return 0

        # 默认图片
        return 1


# ==================== 具体产线实现 ====================


class OcrPipeline(PipelineHandler):
    """OCR产线 - 专注纯文本提取"""

    def get_pipeline_name(self) -> str:
        return "OCR"

    def get_engine_name(self) -> str:
        return "ocr"

    def _parse_api_result(self, api_result: Dict) -> Dict:
        """解析星河API OCR结果"""
        ocr_results = api_result["ocrResults"]
        if not ocr_results:
            return {
                "text": "",
                "confidence": 0,
                "blocks": [],
                "text_type": "api",
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
        }

    def _parse_local_result(self, raw_result) -> Dict:
        """解析本地OCR结果"""
        if not raw_result or not raw_result[0]:
            return {
                "text": "",
                "confidence": 0,
                "blocks": [],
                "text_type": "local",
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
        }

    def _format_output(self, result: Dict, detailed: bool) -> str:
        """格式化OCR输出 - L1核心信息，L2完整数据"""
        if not result["text"].strip():
            return (
                "❌ No text detected"
                if not detailed
                else json.dumps({"error": "No text detected"}, ensure_ascii=False)
            )

        if detailed:
            # L2: 返回所有
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

    async def _log_completion_stats(self, ctx: Context, result: Dict):
        """记录OCR完成统计"""
        text_length = len(result["text"])
        block_count = len(result["blocks"])
        await ctx.info(
            f"OCR completed: {text_length} characters, {block_count} text blocks"
        )


class StructurePipeline(PipelineHandler):
    """结构分析产线 - 专注文档结构和相关图片"""

    def get_pipeline_name(self) -> str:
        return "Structure analysis"

    def get_engine_name(self) -> str:
        return "structure"

    def _parse_api_result(self, api_result: Dict) -> Dict:
        """解析星河API结构结果"""
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
        referenced_images = []

        for i, res in enumerate(layout_results):
            markdown_data = res["markdown"]
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
                    page_referenced_images = (
                        self._extract_referenced_images_from_markdown(
                            text, markdown_data["images"]
                        )
                    )
                    referenced_images.extend(page_referenced_images)

                page = {
                    "page": i,
                    "content": text,
                    "has_images": bool(page_images),
                    "images": page_images,
                }
                pages.append(page)

        return {
            "markdown": "\n".join(markdown_parts),
            "pages": pages,
            "has_images": bool(all_images),
            "images": all_images,
            "referenced_images": referenced_images,
        }

    def _parse_local_result(self, raw_results) -> Dict:
        """解析本地结构结果"""
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

    def _format_output(self, result: Dict, detailed: bool) -> Union[str, List]:
        """格式化结构分析输出 - L1/L2都包含图片"""
        if not result["markdown"].strip():
            return (
                "❌ No document structure detected"
                if not detailed
                else json.dumps({"error": "No structure detected"}, ensure_ascii=False)
            )

        # 检查是否需要混合内容传输（API模式 + 有引用图片）
        if (
            result.get("referenced_images")
            and self.engines.is_api_mode()
            and not detailed
        ):
            try:
                content_list = []

                # 添加文本内容（无图片引用避免重复）
                text_content = self._format_text_only(result, include_image_refs=False)
                content_list.append(text_content)

                # 添加所有在markdown中引用的图片
                for target_image in result["referenced_images"]:
                    image_content = self._process_image_for_transmission(target_image)
                    if image_content:
                        content_list.append(image_content)

                return content_list
            except Exception as e:
                raise RuntimeError(f"Failed to process mixed content: {str(e)}") from e

        # 标准文本返回
        if detailed:
            # L2: 移除大体积图片数据，只保留元数据
            cleaned_result = {k: v for k, v in result.items() if k != "images"}
            return json.dumps(cleaned_result, ensure_ascii=False, indent=2)
        else:
            # L1: 纯markdown + 图片引用（如果有）
            return self._format_text_only(result, include_image_refs=True)

    def _format_text_only(self, result: Dict, include_image_refs: bool = True) -> str:
        """格式化纯文本输出"""
        markdown = result["markdown"]

        if result["images"] and include_image_refs:
            image_refs = "\n\n📸 **Images**: " + ", ".join(
                f"[img{i+1}]({url})" for i, url in enumerate(result["images"])
            )
            markdown += image_refs

        return markdown

    def _process_image_for_transmission(
        self, target_image: str
    ) -> Optional[FastMCPImage]:
        """处理图片用于传输 - 使用filetype库进行robust格式检测"""
        try:
            if target_image.startswith(("http://", "https://")):
                # URL：下载图片
                import asyncio

                async def download_image():
                    async with httpx.AsyncClient() as client:
                        response = await client.get(target_image)
                        if response.status_code == 200:
                            # 使用filetype检测实际格式
                            image_data = response.content
                            detected_type = filetype.guess(image_data)
                            if detected_type and detected_type.mime.startswith(
                                "image/"
                            ):
                                format_type = detected_type.extension
                            else:
                                # 从Content-Type头部获取格式作为fallback
                                format_type = response.headers.get(
                                    "content-type", "image/jpeg"
                                ).split("/")[-1]
                            return FastMCPImage(data=image_data, format=format_type)
                    return None

                # 在同步上下文中运行异步函数
                try:
                    loop = asyncio.get_running_loop()
                    # 这里需要处理异步，但由于混合内容处理的复杂性，暂时简化
                    return None
                except RuntimeError:
                    return None
            else:
                # 假设为base64数据：使用filetype进行robust检测
                image_data = base64.b64decode(target_image)

                # 使用filetype库进行格式检测
                detected_type = filetype.guess(image_data)
                if detected_type and detected_type.mime.startswith("image/"):
                    format_type = detected_type.extension
                else:
                    # 如果filetype无法识别，使用默认格式
                    format_type = "jpeg"

                return FastMCPImage(data=image_data, format=format_type)
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(
                f"Image processing failed for {target_image[:50]}...: {e}"
            )
            return None

    def _extract_referenced_images_from_markdown(
        self, markdown_text: str, available_images: Dict[str, str]
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

    async def _log_completion_stats(self, ctx: Context, result: Dict):
        """记录结构分析完成统计"""
        page_count = len(result["pages"])
        image_count = len(result["images"])
        await ctx.info(
            f"Structure analysis completed: {page_count} pages, {image_count} images"
        )
