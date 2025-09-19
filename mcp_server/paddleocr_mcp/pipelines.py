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

# TODO:
# 1. Reuse `httpx` client.
# 2. Use `contextvars` to manage MCP context objects.
# 3. Implement structured logging, log stack traces, and log operation timing.
# 4. Report progress for long-running operations.

import abc
import asyncio
import base64
import io
import json
import re
from pathlib import PurePath
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, List, NoReturn, Optional, Type, Union
from urllib.parse import urlparse

import httpx
import numpy as np
import puremagic
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


def _infer_file_type_from_bytes(data: bytes) -> Optional[str]:
    mime = puremagic.from_string(data, mime=True)
    if mime.startswith("image/"):
        return "image"
    elif mime == "application/pdf":
        return "pdf"
    return None


def get_str_with_max_len(obj: object, max_len: int) -> str:
    s = str(obj)
    if len(s) > max_len:
        return s[:max_len] + "..."
    else:
        return s


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
            self._closed = True

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
            timeout: Read timeout in seconds for HTTP requests.
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
        self._timeout = timeout or 60

        if self._mode == "local":
            if not LOCAL_OCR_AVAILABLE:
                raise RuntimeError("PaddleOCR is not locally available")
            try:
                self._engine = self._create_local_engine()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create PaddleOCR engine: {str(e)}"
                ) from e

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

    @abc.abstractmethod
    def _get_service_endpoint(self) -> str:
        """Get the service endpoint.

        Returns:
            Service endpoint path.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _transform_local_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform keyword arguments for local execution.

        Args:
            kwargs: Keyword arguments.

        Returns:
            Transformed keyword arguments.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _transform_service_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform keyword arguments for service execution.

        Args:
            kwargs: Keyword arguments.

        Returns:
            Transformed keyword arguments.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def _parse_local_result(
        self, local_result: Dict, ctx: Context
    ) -> Dict[str, Any]:
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
    async def _format_output(
        self,
        result: Dict[str, Any],
        detailed: bool,
        ctx: Context,
        **kwargs: Any,
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        """Format output into simple or detailed format.

        Args:
            result: Processing result.
            detailed: Whether to use detailed format.
            ctx: MCP context.
            **kwargs: Additional arguments.

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


class SimpleInferencePipelineHandler(PipelineHandler):
    """Base class for simple inference pipeline handlers."""

    async def process(
        self,
        input_data: str,
        output_mode: OutputMode,
        ctx: Context,
        file_type: Optional[str] = None,
        infer_kwargs: Optional[Dict[str, Any]] = None,
        format_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        """Process input data through the pipeline.

        Args:
            input_data: Input data (file path, URL, or Base64).
            output_mode: Output mode ("simple" or "detailed").
            ctx: MCP context.
            file_type: File type for URLs ("image", "pdf", or None for auto-detection).
            infer_kwargs: Additional arguments for performing pipeline inference.
            format_kwargs: Additional arguments for formatting the output.

        Returns:
            Processed result in the requested output format.
        """
        infer_kwargs = infer_kwargs or {}
        format_kwargs = format_kwargs or {}
        try:
            await ctx.info(
                f"Starting {self._pipeline} processing (source: {self._ppocr_source})"
            )

            if self._mode == "local":
                processed_input = self._process_input_for_local(input_data, file_type)
                infer_kwargs = self._transform_local_kwargs(infer_kwargs)
                raw_result = await self._predict_with_local_engine(
                    processed_input, ctx, **infer_kwargs
                )
                result = await self._parse_local_result(raw_result, ctx)
            else:
                processed_input, inferred_file_type = self._process_input_for_service(
                    input_data, file_type
                )
                infer_kwargs = self._transform_service_kwargs(infer_kwargs)
                raw_result = await self._call_service(
                    processed_input, inferred_file_type, ctx, **infer_kwargs
                )
                result = await self._parse_service_result(raw_result, ctx)

            await self._log_completion_stats(result, ctx)
            return await self._format_output(
                result, output_mode == "detailed", ctx, **format_kwargs
            )

        except Exception as e:
            await ctx.error(f"{self._pipeline} processing failed: {str(e)}")
            self._handle_error(e, output_mode)

    def _process_input_for_local(
        self, input_data: str, file_type: Optional[str]
    ) -> Union[str, np.ndarray]:
        # TODO: Use `file_type` to handle more cases.
        if _is_base64(input_data):
            if input_data.startswith("data:"):
                base64_data = input_data.split(",", 1)[1]
            else:
                base64_data = input_data
            try:
                image_bytes = base64.b64decode(base64_data)
                file_type = _infer_file_type_from_bytes(image_bytes)
                if file_type != "image":
                    raise ValueError("Currently, only images can be passed via Base64.")
                image_pil = PILImage.open(io.BytesIO(image_bytes))
                image_arr = np.array(image_pil.convert("RGB"))
                return np.ascontiguousarray(image_arr[..., ::-1])
            except Exception as e:
                raise ValueError(f"Failed to decode Base64 image: {str(e)}") from e
        elif _is_file_path(input_data) or _is_url(input_data):
            return input_data
        else:
            raise ValueError("Invalid input data format")

    def _process_input_for_service(
        self, input_data: str, file_type: Optional[str]
    ) -> tuple[str, Optional[str]]:
        if _is_url(input_data):
            norm_ft = None
            if isinstance(file_type, str):
                if file_type.lower() in ("None", "none", "null", "unknown", ""):
                    norm_ft = None
                else:
                    norm_ft = file_type.lower()
            return input_data, norm_ft
        elif _is_base64(input_data):
            try:
                if input_data.startswith("data:"):
                    base64_data = input_data.split(",", 1)[1]
                else:
                    base64_data = input_data
                bytes_ = base64.b64decode(base64_data)
                file_type_str = _infer_file_type_from_bytes(bytes_)
                if file_type_str is None:
                    raise ValueError(
                        "Unsupported file type in Base64 data. "
                        "Only images (JPEG, PNG, etc.) and PDF documents are supported."
                    )
                return input_data, file_type_str
            except Exception as e:
                raise ValueError(f"Failed to decode Base64 data: {str(e)}") from e
        elif _is_file_path(input_data):
            try:
                with open(input_data, "rb") as f:
                    bytes_ = f.read()
                input_data = base64.b64encode(bytes_).decode("ascii")
                file_type_str = _infer_file_type_from_bytes(bytes_)
                if file_type_str is None:
                    raise ValueError(
                        f"Unsupported file type for '{input_data}'. "
                        "Only images (JPEG, PNG, etc.) and PDF documents are supported."
                    )
                return input_data, file_type_str
            except Exception as e:
                raise ValueError(f"Failed to read file: {str(e)}") from e
        else:
            raise ValueError("Invalid input data format")

    async def _call_service(
        self,
        processed_input: str,
        file_type: Optional[str],
        ctx: Context,
        **kwargs: Any,
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
            timeout = httpx.Timeout(
                connect=30.0, read=self._timeout, write=30.0, pool=30.0
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"HTTP request failed: {type(e).__name__}: {str(e)}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid service response: {str(e)}")

    def _prepare_service_payload(
        self, processed_input: str, file_type: Optional[str], **kwargs: Any
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"file": processed_input, **kwargs}
        if file_type == "image":
            payload["fileType"] = 1
        elif file_type == "pdf":
            payload["fileType"] = 0
        else:
            payload["fileType"] = None

        return payload

    def _handle_error(self, exc: Exception, output_mode: OutputMode) -> NoReturn:
        raise exc


class OCRHandler(SimpleInferencePipelineHandler):
    def register_tools(self, mcp: FastMCP) -> None:
        @mcp.tool("ocr")
        async def _ocr(
            input_data: str,
            output_mode: OutputMode = "simple",
            file_type: Optional[str] = None,
            *,
            ctx: Context,
        ) -> Union[str, List[Union[TextContent, ImageContent]]]:
            """Extracts text from images and PDFs. Accepts file path, URL, or Base64.

            Args:
                input_data: The file to process (file path, URL, or Base64 string).
                output_mode: The desired output format.
                    - "simple": (Default) Clean, readable text suitable for most use cases.
                    - "detailed": A JSON output including text, confidence, and precise bounding box coordinates. Only use this when coordinates are specifically required.
                file_type: File type. This parameter is REQUIRED when `input_data` is a URL and should be omitted for other types.
                    - "image": For image files
                    - "pdf": For PDF documents
                    - None: For unknown file types
            """
            await ctx.info(
                f"--- OCR tool received `input_data`: {get_str_with_max_len(input_data, 50)} ---"
            )
            return await self.process(input_data, output_mode, ctx, file_type)

    def _create_local_engine(self) -> Any:
        return PaddleOCR(
            paddlex_config=self._pipeline_config,
            device=self._device,
        )

    def _get_service_endpoint(self) -> str:
        return "ocr"

    def _transform_local_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "use_doc_unwarping": False,
            "use_doc_orientation_classify": False,
        }

    def _transform_service_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "useDocUnwarping": False,
            "useDocOrientationClassify": False,
        }

    async def _parse_local_result(self, local_result: Dict, ctx: Context) -> Dict:
        result = local_result[0]
        texts = result["rec_texts"]
        scores = result["rec_scores"]
        boxes = result["rec_boxes"]

        clean_texts, confidences, text_lines = [], [], []

        for i, text in enumerate(texts):
            if text and text.strip():
                conf = scores[i] if i < len(scores) else 0
                clean_texts.append(text.strip())
                confidences.append(conf)
                instance = {
                    "text": text.strip(),
                    "confidence": round(conf, 3),
                    "bbox": boxes[i].tolist(),
                }
                text_lines.append(instance)

        return {
            "text": "\n".join(clean_texts),
            "confidence": sum(confidences) / len(confidences) if confidences else 0,
            "text_lines": text_lines,
        }

    async def _parse_service_result(self, service_result: Dict, ctx: Context) -> Dict:
        result_data = service_result.get("result", service_result)
        ocr_results = result_data.get("ocrResults")

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
                    instance = {
                        "text": text.strip(),
                        "confidence": round(conf, 3),
                        "bbox": boxes[i],
                    }
                    text_lines.append(instance)

        return {
            "text": "\n".join(all_texts),
            "confidence": (
                sum(all_confidences) / len(all_confidences) if all_confidences else 0
            ),
            "text_lines": text_lines,
        }

    async def _log_completion_stats(self, result: Dict, ctx: Context) -> None:
        text_length = len(result["text"])
        text_line_count = len(result["text_lines"])
        await ctx.info(
            f"OCR completed: {text_length} characters, {text_line_count} text lines"
        )

    async def _format_output(
        self,
        result: Dict,
        detailed: bool,
        ctx: Context,
        **kwargs: Any,
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        if not result["text"].strip():
            return (
                "❌ No text detected"
                if not detailed
                else json.dumps({"error": "No text detected"}, ensure_ascii=False)
            )

        if detailed:
            return json.dumps(result, ensure_ascii=False, indent=2)
        else:
            confidence = result["confidence"]
            text_line_count = len(result["text_lines"])

            output = result["text"]
            if confidence > 0:
                output += f"\n\n📊 Confidence: {(confidence * 100):.1f}% | {text_line_count} text lines"

            return output


class PPStructureV3Handler(SimpleInferencePipelineHandler):
    def register_tools(self, mcp: FastMCP) -> None:
        @mcp.tool("pp_structurev3")
        async def _pp_structurev3(
            input_data: str,
            output_mode: OutputMode = "simple",
            file_type: Optional[str] = None,
            return_images: bool = True,
            *,
            ctx: Context,
        ) -> Union[str, List[Union[TextContent, ImageContent]]]:
            """Extracts structured markdown from complex documents (images/PDFs), including tables, formulas, etc. Accepts file path, URL, or Base64.

            Args:
                input_data: The file to process (file path, URL, or Base64 string).
                output_mode: The desired output format.
                    - "simple": (Default) Clean, readable markdown with embedded images. Best for most use cases.
                    - "detailed": JSON data about document structure, plus markdown. Only use this when coordinates are specifically required.
                file_type: File type. This parameter is REQUIRED when `input_data` is a URL and should be omitted for other types.
                    - "image": For image files
                    - "pdf": For PDF documents
                    - None: For unknown file types
                return_images: Whether to return the images extracted from the document.
            """
            return await self.process(
                input_data,
                output_mode,
                ctx,
                file_type,
                format_kwargs={"return_images": return_images},
            )

    def _create_local_engine(self) -> Any:
        return PPStructureV3(
            paddlex_config=self._pipeline_config,
            device=self._device,
        )

    def _get_service_endpoint(self) -> str:
        return "layout-parsing"

    def _transform_local_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "use_doc_unwarping": False,
            "use_doc_orientation_classify": False,
        }

    def _transform_service_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "useDocUnwarping": False,
            "useDocOrientationClassify": False,
        }

    async def _parse_local_result(self, local_result: Dict, ctx: Context) -> Dict:
        markdown_parts = []
        all_images_mapping = {}
        detailed_results = []

        for result in local_result:
            markdown = result.markdown
            text = markdown["markdown_texts"]
            markdown_parts.append(text)
            images = markdown["markdown_images"]
            processed_images = {}
            for img_key, img_data in images.items():
                with io.BytesIO() as buffer:
                    img_data.save(buffer, format="JPEG")
                    processed_images[img_key] = base64.b64encode(buffer.getvalue())
            all_images_mapping.update(processed_images)
            detailed_results.append(result)

        return {
            # TODO: Page concatenation can be done better via `pipeline.concatenate_markdown_pages`
            "markdown": "\n".join(markdown_parts),
            "pages": len(local_result),
            "images_mapping": all_images_mapping,
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

        markdown_parts = []
        all_images_mapping = {}
        detailed_results = []

        for res in layout_results:
            markdown_parts.append(res["markdown"]["text"])
            images = res["markdown"]["images"]
            processed_images = {}
            for img_key, img_data in images.items():
                processed_images[img_key] = await self._process_image_data(
                    img_data, ctx
                )
            all_images_mapping.update(processed_images)
            detailed_results.append(res["prunedResult"])

        return {
            "markdown": "\n".join(markdown_parts),
            "pages": len(layout_results),
            "images_mapping": all_images_mapping,
            "detailed_results": detailed_results,
        }

    async def _process_image_data(self, img_data: str, ctx: Context) -> str:
        if _is_url(img_data):
            try:
                timeout = httpx.Timeout(connect=30.0, read=30.0, write=30.0, pool=30.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(img_data)
                    response.raise_for_status()
                    img_bytes = response.content
                    return base64.b64encode(img_bytes).decode("ascii")
            except Exception as e:
                await ctx.error(
                    f"Failed to download image from URL {img_data}: {str(e)}"
                )
                return img_data
        elif _is_base64(img_data):
            return img_data
        else:
            await ctx.error(
                f"Unknown image data format: {get_str_with_max_len(img_data, 50)}"
            )
            return img_data

    async def _log_completion_stats(self, result: Dict, ctx: Context) -> None:
        page_count = result["pages"]
        await ctx.info(f"Layout parsing completed: {page_count} pages")

    async def _format_output(
        self,
        result: Dict,
        detailed: bool,
        ctx: Context,
        **kwargs: Any,
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        if not result["markdown"].strip():
            return (
                "❌ No document content detected"
                if not detailed
                else json.dumps({"error": "No content detected"}, ensure_ascii=False)
            )

        markdown_text = result["markdown"]
        images_mapping = result.get("images_mapping", {})

        if kwargs.get("return_images"):
            content_list = self._parse_markdown_with_images(
                markdown_text, images_mapping
            )
        else:
            content_list = [TextContent(type="text", text=markdown_text)]

        if detailed:
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

        return content_list

    def _parse_markdown_with_images(
        self, markdown_text: str, images_mapping: Dict[str, str]
    ) -> List[Union[TextContent, ImageContent]]:
        """Parse markdown text and return mixed list of text and images."""
        if not images_mapping:
            return [TextContent(type="text", text=markdown_text)]

        content_list = []
        img_pattern = r'<img[^>]+src="([^"]+)"[^>]*>'
        last_pos = 0

        for match in re.finditer(img_pattern, markdown_text):
            text_before = markdown_text[last_pos : match.start()]
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
