# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
import base64
import io
import json
import mimetypes
import re
from pathlib import PurePath
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Type, Union
from urllib.parse import urlparse

import httpx
import magic
import numpy as np
from fastmcp import Context, FastMCP
from mcp.types import ImageContent, TextContent
from PIL import Image as PILImage
from typing_extensions import Literal, Self, assert_never

try:
    from paddleocr import PaddleOCR, PPStructureV3

    LOCAL_OCR_AVAILABLE = True
except ImportError:
    LOCAL_OCR_AVAILABLE = False


OutputMode = Literal["simple", "detailed"]


def _is_file_path(s: str) -> bool:
    try:
        PurePath(s)
        return True
    except Exception:
        return False


def _is_base64(s: str) -> bool:
    pattern = r"^[A-Za-z0-9+/]+={0,2}$"
    return bool(re.fullmatch(pattern, s))


def _is_url(s: str) -> bool:
    if not (s.startswith("http://") or s.startswith("https://")):
        return False
    result = urlparse(s)
    return all([result.scheme, result.netloc]) and result.scheme in ("http", "https")


def _infer_file_type_from_url(url: str) -> str:
    url_parts = urlparse(url)
    filename = url_parts.path.split("/")[-1]
    file_type = mimetypes.guess_type(filename)[0]
    if not file_type:
        return "UNKNOWN"
    if file_type.startswith("image/"):
        return "IMAGE"
    elif file_type == "application/pdf":
        return "PDF"
    return "UNKNOWN"


def _infer_file_type_from_bytes(data: bytes) -> str:
    mime = magic.from_buffer(data, mime=True)
    if mime.startswith("image/"):
        return "IMAGE"
    elif mime == "application/pdf":
        return "PDF"
    return "UNKNOWN"


class _EngineWrapper:
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

    def _worker(self) -> None:
        while not self._closed:
            item = self._queue.get()
            if item is None:
                break
            func, args, kwargs, fut = item
            try:
                result = func(*args, **kwargs)
                self._loop.call_soon_threadsafe(fut.set_result, result)
            except Exception as e:
                self._loop.call_soon_threadsafe(fut.set_exception, e)
            finally:
                self._queue.task_done()


class PipelineHandler(abc.ABC):
    """Abstract base class for pipeline handlers."""

    def __init__(
        self,
        pipeline: str,
        ppocr_source: str,
        pipeline_config: Optional[str],
        device: Optional[str],
        server_url: Optional[str],
        aistudio_access_token: Optional[str],
        timeout: Optional[int],
    ) -> None:
        """Initialize the pipeline handler.

        Args:
            pipeline: Pipeline name.
            ppocr_source: Source of PaddleOCR functionality.
            pipeline_config: Path to pipeline configuration.
            device: Device to run inference on.
            server_url: Base URL for service mode.
            aistudio_access_token: AI Studio access token.
            timeout: Timeout in seconds.
        """
        self._pipeline = pipeline
        if ppocr_source == "local":
            self._mode = "local"
        elif ppocr_source in ("aistudio", "self_hosted"):
            self._mode = "service"
        else:
            raise ValueError(f"Unknown PaddleOCR source {repr(ppocr_source)}")
        self._ppocr_source = ppocr_source
        self._pipeline_config = pipeline_config
        self._device = device
        self._server_url = server_url
        self._aistudio_access_token = aistudio_access_token
        self._timeout = timeout or 30  # Default timeout of 30 seconds

        if self._mode == "local":
            if not LOCAL_OCR_AVAILABLE:
                raise RuntimeError("PaddleOCR is not locally available")
            self._engine = self._create_local_engine()

        self._status: Literal["initialized", "started", "stopped"] = "initialized"

    async def start(self) -> None:
        if self._status == "initialized":
            if self._mode == "local":
                self._engine_wrapper = _EngineWrapper(self._engine)
            self._status = "started"
        elif self._status == "started":
            pass
        elif self._status == "stopped":
            raise RuntimeError("Pipeline handler has already been stopped")
        else:
            assert_never(self._status)

    async def stop(self) -> None:
        if self._status == "initialized":
            raise RuntimeError("Pipeline handler has not been started")
        elif self._status == "started":
            if self._mode == "local":
                await self._engine_wrapper.close()
            self._status = "stopped"
        elif self._status == "stopped":
            pass
        else:
            assert_never(self._status)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        await self.stop()

    @abc.abstractmethod
    def register_tools(self, mcp: FastMCP) -> None:
        """Register tools with the MCP server.

        Args:
            mcp: The `FastMCP` instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _create_local_engine(self) -> Any:
        """Create the local OCR engine.

        Returns:
            The OCR engine instance.
        """
        raise NotImplementedError


class SimpleInferencePipelineHandler(PipelineHandler):
    """Base class for simple inference pipeline handlers."""

    async def process(
        self, input_data: str, output_mode: OutputMode, ctx: Context, **kwargs: Any
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        """Process input data through the pipeline.

        Args:
            input_data: Input data (file path, URL, or Base64).
            output_mode: Output mode ("simple" or "detailed").
            ctx: MCP context.
            **kwargs: Additional pipeline-specific arguments.

        Returns:
            Processed result in the requested output format.
        """
        try:
            await ctx.info(
                f"Starting {self._pipeline} processing (source: {self._ppocr_source})"
            )

            if self._mode == "local":
                processed_input = self._process_input_for_local(input_data)
                raw_result = await self._predict_with_local_engine(
                    processed_input, ctx, **kwargs
                )
                result = self._parse_local_result(raw_result, ctx)
            else:
                processed_input, file_type = self._process_input_for_service(input_data)
                raw_result = await self._call_service(
                    processed_input, file_type, ctx, **kwargs
                )
                result = await self._parse_service_result(raw_result, ctx)

            await self._log_completion_stats(result, ctx)
            return self._format_output(result, output_mode == "detailed", ctx)

        except Exception as e:
            await ctx.error(f"{self._pipeline} processing failed: {str(e)}")
            return self._handle_error(str(e), output_mode)

    def _process_input_for_local(self, input_data: str) -> Union[str, np.ndarray]:
        if _is_file_path(input_data) or _is_url(input_data):
            return input_data
        elif _is_base64(input_data):
            if input_data.startswith("data:"):
                base64_data = input_data.split(",", 1)[1]
            else:
                base64_data = input_data
            try:
                image_bytes = base64.b64decode(base64_data)
                image_pil = PILImage.open(io.BytesIO(image_bytes))
                image_arr = np.array(image_pil.convert("RGB"))
                # Convert RGB to BGR
                return np.ascontiguousarray(image_arr[..., ::-1])
            except Exception as e:
                raise ValueError(f"Failed to decode Base64 image: {e}")
        else:
            raise ValueError("Invalid input data format")

    def _process_input_for_service(self, input_data: str) -> tuple[str, str]:
        if _is_file_path(input_data):
            try:
                with open(input_data, "rb") as f:
                    bytes_ = f.read()
                input_data = base64.b64encode(bytes_).decode("ascii")
                file_type = _infer_file_type_from_bytes(bytes_)
            except Exception as e:
                raise ValueError(f"Failed to read file: {e}")
        elif _is_url(input_data):
            file_type = _infer_file_type_from_url(input_data)
        elif _is_base64(input_data):
            try:
                if input_data.startswith("data:"):
                    base64_data = input_data.split(",", 1)[1]
                else:
                    base64_data = input_data
                bytes_ = base64.b64decode(base64_data)
                file_type = _infer_file_type_from_bytes(bytes_)
            except Exception as e:
                raise ValueError(f"Failed to decode Base64 data: {e}")
        else:
            raise ValueError("Invalid input data format")

        return input_data, file_type

    async def _call_service(
        self, processed_input: str, file_type: str, ctx: Context, **kwargs: Any
    ) -> Dict[str, Any]:
        if not self._server_url:
            raise RuntimeError("Server URL not configured")

        endpoint = self._get_service_endpoint()
        url = f"{self._server_url.rstrip('/')}/{endpoint.lstrip('/')}"

        payload = self._prepare_service_payload(processed_input, file_type, **kwargs)
        headers = {"Content-Type": "application/json"}

        if self._ppocr_source == "aistudio":
            if not self._aistudio_access_token:
                raise RuntimeError("Missing AI Studio access token")
            headers["Authorization"] = f"token {self._aistudio_access_token}"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Service call failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid service response: {str(e)}")

    def _prepare_service_payload(
        self, processed_input: str, file_type: str, **kwargs: Any
    ) -> Dict[str, Any]:
        api_file_type = 1 if file_type == "IMAGE" else 0
        payload = {"file": processed_input, "fileType": api_file_type, **kwargs}
        return payload

    def _handle_error(
        self, error_msg: str, output_mode: OutputMode
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        if output_mode == "detailed":
            return [TextContent(type="text", text=f"Error: {error_msg}")]
        return f"Error: {error_msg}"

    @abc.abstractmethod
    def _get_service_endpoint(self) -> str:
        """Get the service endpoint.

        Returns:
            Service endpoint path.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _parse_local_result(self, local_result: Dict, ctx: Context) -> Dict[str, Any]:
        """Parse raw result from local engine into a unified format.

        Args:
            local_result: Raw result from local engine.
            ctx: MCP context.

        Returns:
            Parsed result in unified format.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def _parse_service_result(
        self, service_result: Dict[str, Any], ctx: Context
    ) -> Dict[str, Any]:
        """Parse raw result from the service into a unified format.

        Args:
            service_result: Raw result from the service.
            ctx: MCP context.

        Returns:
            Parsed result in unified format.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def _log_completion_stats(self, result: Dict[str, Any], ctx: Context) -> None:
        """Log statistics after processing completion.

        Args:
            result: Processing result.
            ctx: MCP context.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _format_output(
        self, result: Dict[str, Any], detailed: bool, ctx: Context
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        """Format output into simple or detailed format.

        Args:
            result: Processing result.
            detailed: Whether to use detailed format.
            ctx: MCP context.

        Returns:
            Formatted output in requested format.
        """
        raise NotImplementedError

    async def _predict_with_local_engine(
        self, processed_input: Union[str, np.ndarray], ctx: Context, **kwargs: Any
    ) -> Dict:
        if not hasattr(self, "_engine_wrapper"):
            raise RuntimeError("Engine wrapper has not been initialized")
        return await self._engine_wrapper.call(
            self._engine_wrapper.engine.predict, processed_input, **kwargs
        )


class OCRHandler(SimpleInferencePipelineHandler):
    def register_tools(self, mcp: FastMCP) -> None:
        @mcp.tool()
        async def _ocr(
            input_data: str,
            output_mode: OutputMode,
            ctx: Context,
        ) -> Union[str, List[Union[TextContent, ImageContent]]]:
            """Extract text from images and PDFs.

            Args:
                input_data: File path, URL, or Base64 data.
                output_mode: "simple" for clean text, "detailed" for JSON with positioning.
            """
            return await self.process(input_data, output_mode, ctx)

    def _create_local_engine(self) -> Any:
        return PaddleOCR(
            paddlex_config=self._pipeline_config,
            device=self._device,
            enable_mkldnn=False,
        )

    def _get_service_endpoint(self) -> str:
        return "ocr"

    def _parse_local_result(self, local_result: Dict, ctx: Context) -> Dict:
        result = local_result[0]
        texts = result["rec_texts"]
        scores = result["rec_scores"]
        boxes = result["rec_boxes"]

        # Direct assembly
        clean_texts, confidences, blocks = [], [], []

        for i, text in enumerate(texts):
            if text and text.strip():
                conf = scores[i] if i < len(scores) else 0
                clean_texts.append(text.strip())
                confidences.append(conf)
                block = {
                    "text": text.strip(),
                    "confidence": round(conf, 3),
                    "bbox": boxes[i].tolist(),
                }
                blocks.append(block)

        return {
            "text": "\n".join(clean_texts),
            "confidence": sum(confidences) / len(confidences) if confidences else 0,
            "blocks": blocks,
        }

    async def _parse_service_result(self, service_result: Dict, ctx: Context) -> Dict:
        result_data = service_result.get("result", service_result)
        ocr_results = result_data.get("ocrResults")

        # Direct extraction and assembly
        all_texts, all_confidences, blocks = [], [], []

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
                    block = {
                        "text": text.strip(),
                        "confidence": round(conf, 3),
                        "bbox": boxes[i],
                    }
                    blocks.append(block)

        return {
            "text": "\n".join(all_texts),
            "confidence": (
                sum(all_confidences) / len(all_confidences) if all_confidences else 0
            ),
            "blocks": blocks,
        }

    async def _log_completion_stats(self, result: Dict, ctx: Context) -> None:
        text_length = len(result["text"])
        block_count = len(result["blocks"])
        await ctx.info(
            f"OCR completed: {text_length} characters, {block_count} text blocks"
        )

    def _format_output(
        self, result: Dict, detailed: bool, ctx: Context
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        if not result["text"].strip():
            return (
                "❌ No text detected"
                if not detailed
                else json.dumps({"error": "No text detected"}, ensure_ascii=False)
            )

        if detailed:
            # L2: Return all data
            return json.dumps(result, ensure_ascii=False, indent=2)
        else:
            # L1: Core text + key statistics
            confidence = result["confidence"]
            block_count = len(result["blocks"])

            output = result["text"]
            if confidence > 0:
                output += f"\n\n📊 Confidence: {(confidence * 100):.1f}% | {block_count} text blocks"

            return output


class PPStructureV3Handler(SimpleInferencePipelineHandler):
    def register_tools(self, mcp: FastMCP) -> None:
        @mcp.tool()
        async def _pp_structurev3(
            input_data: str,
            output_mode: OutputMode,
            ctx: Context,
        ) -> Union[str, List[Union[TextContent, ImageContent]]]:
            """Document layout analysis.

            Args:
                input_data: File path, URL, or Base64 data.
                output_mode: "simple" for markdown text, "detailed" for JSON with metadata + prunedResult.

            Returns:
                - Simple: Markdown text + images (if available)
                - Detailed: prunedResult/local detailed info + markdown text + images
            """
            return await self.process(input_data, output_mode, ctx)

    def _create_local_engine(self) -> Any:
        return PPStructureV3(paddlex_config=self._pipeline_config, device=self._device)

    def _get_service_endpoint(self) -> str:
        return "layout-parsing"

    def _parse_local_result(self, local_result: Dict, ctx: Context) -> Dict:
        markdown_parts = []
        detailed_results = []

        # TODO return images
        for result in local_result:
            text = result.markdown["markdown_texts"]
            markdown_parts.append(text)
            detailed_results.append(result)

        return {
            # TODO: Page concatenation can be done better via `pipeline.concatenate_markdown_pages`
            "markdown": "\n".join(markdown_parts),
            "pages": len(local_result),
            "images_mapping": {},
            "detailed_results": detailed_results,
        }

    async def _parse_service_result(self, service_result: Dict, ctx: Context) -> Dict:
        result_data = service_result.get("result", service_result)
        layout_results = result_data.get("layoutParsingResults")

        if not layout_results:
            return {
                "markdown": "",
                "pages": 0,
                "images_mapping": {},
                "detailed_results": [],
            }

        # 简化：直接提取需要的信息
        markdown_parts = []
        all_images_mapping = {}
        detailed_results = []

        for res in layout_results:
            # 提取markdown文本
            markdown_parts.append(res["markdown"]["text"])
            # 提取图片
            all_images_mapping.update(res["markdown"]["images"])
            # 保存prunedResult用于L2详细信息
            detailed_results.append(res["prunedResult"])

        return {
            "markdown": "\n".join(markdown_parts),
            "pages": len(layout_results),  # 简化为页数
            "images_mapping": all_images_mapping,
            "detailed_results": detailed_results,
        }

    async def _log_completion_stats(self, result: Dict, ctx: Context) -> None:
        page_count = result["pages"]  # 现在是数字而不是列表
        await ctx.info(f"Structure analysis completed: {page_count} pages")

    def _format_output(
        self, result: Dict, detailed: bool, ctx: Context
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        if not result["markdown"].strip():
            return (
                "❌ No document content detected"
                if not detailed
                else json.dumps({"error": "No content detected"}, ensure_ascii=False)
            )

        markdown_text = result["markdown"]
        images_mapping = result.get("images_mapping", {})

        if detailed:
            # L2: 返回统一的详细结果 + markdown混合内容
            content_list = []
            if "detailed_results" in result and result["detailed_results"]:
                for detailed_result in result["detailed_results"]:
                    content_list.append(
                        TextContent(
                            type="text",
                            text=json.dumps(
                                detailed_result,
                                ensure_ascii=False,
                                indent=2,
                                default=str,
                            ),
                        )
                    )

            # 添加markdown混合内容
            content_list.extend(
                self._parse_markdown_with_images(markdown_text, images_mapping)
            )

            return content_list
        else:
            # L1: 简化的混合内容格式，只包含markdown和图片
            return self._parse_markdown_with_images(markdown_text, images_mapping)

    def _parse_markdown_with_images(
        self, markdown_text: str, images_mapping: Dict[str, str]
    ) -> List[Union[TextContent, ImageContent]]:
        """解析markdown文本，返回文字和图片的混合列表"""
        if not images_mapping:
            # 没有图片，直接返回文本
            return [TextContent(type="text", text=markdown_text)]

        content_list = []
        img_pattern = r'<img[^>]+src="([^"]+)"[^>]*>'
        last_pos = 0

        for match in re.finditer(img_pattern, markdown_text):
            # 添加图片前的文本
            text_before = markdown_text[last_pos : match.start()]
            if text_before.strip():
                content_list.append(TextContent(type="text", text=text_before))

            # 添加图片
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

        # 添加剩余文本
        remaining_text = markdown_text[last_pos:]
        if remaining_text.strip():
            content_list.append(TextContent(type="text", text=remaining_text))

        return content_list or [TextContent(type="text", text=markdown_text)]


_PIPELINE_HANDLERS: Dict[str, Type[PipelineHandler]] = {
    "OCR": OCRHandler,
    "PP-StructureV3": PPStructureV3Handler,
}


def create_pipeline_handler(
    pipeline: str, /, *args: Any, **kwargs: Any
) -> PipelineHandler:
    if pipeline in _PIPELINE_HANDLERS:
        cls = _PIPELINE_HANDLERS[pipeline]
        return cls(pipeline, *args, **kwargs)
    else:
        raise ValueError(f"Unknown pipeline {repr(pipeline)}")
