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
import re
from pathlib import PurePath
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urlparse

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
    """Wrapper to run synchronous PaddleOCR engine in async context.

    Args:
        engine: The synchronous PaddleOCR engine instance to wrap.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._queue: Queue = Queue()
        self._closed = False
        self._loop = asyncio.get_running_loop()
        self._thread = Thread(target=self._worker)
        self._thread.start()

    @property
    def engine(self) -> Any:
        """Get the wrapped engine instance."""
        return self._engine

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Call a function on the wrapped engine asynchronously.

        Args:
            func: The function to call on the engine.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function call.

        Raises:
            RuntimeError: If the engine wrapper has been closed.
        """
        if self._closed:
            raise RuntimeError("Engine wrapper has already been closed")
        fut = self._loop.create_future()
        self._queue.put((func, args, kwargs, fut))
        return await fut

    async def close(self) -> None:
        """Close the engine wrapper and stop the worker thread."""
        if not self._closed:
            self._queue.put(None)
            await self._loop.run_in_executor(None, self._thread.join)
            self._closed = True

    def _worker(self) -> None:
        """Worker thread that processes tasks from the queue."""
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


class LocalExecutor(Executor):
    """Executor for local PaddleOCR inference.

    Args:
        pipeline: The pipeline type to use (e.g., "OCR", "PP-StructureV3").
        pipeline_config: Optional path to pipeline configuration file.
        device: Optional device specification (e.g., "cpu", "gpu:0").
    """

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
        """Create PaddleOCR engine based on pipeline type"""
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
        """Check if a string is a valid file path.

        Args:
            s: The string to check.

        Returns:
            True if the string is a valid file path, False otherwise.
        """
        try:
            PurePath(s)
            return True
        except Exception:
            return False

    def _is_url(self, s: str) -> bool:
        """Check if a string is a valid HTTP/HTTPS URL.

        Args:
            s: The string to check.

        Returns:
            True if the string is a valid HTTP/HTTPS URL, False otherwise.
        """
        if not (s.startswith("http://") or s.startswith("https://")):
            return False

        result = urlparse(s)
        return all([result.scheme, result.netloc]) and result.scheme in (
            "http",
            "https",
        )

    def _is_base64(self, s: str) -> bool:
        """Check if a string is a valid Base64-encoded string.

        Args:
            s: The string to check.

        Returns:
            True if the string is valid Base64, False otherwise.
        """
        pattern = r"^[A-Za-z0-9+/]+={0,2}$"
        return bool(re.fullmatch(pattern, s))

    def _infer_file_type_from_bytes(self, data: bytes) -> Optional[str]:
        """Infer file type from raw bytes using magic numbers.

        Args:
            data: Raw bytes of the file.

        Returns:
            The inferred file type ("image" or "pdf"), or None if unknown.
        """
        mime = puremagic.from_string(data, mime=True)
        if mime.startswith("image/"):
            return "image"
        elif mime == "application/pdf":
            return "pdf"
        return None

    def _process_input_for_local(self, input_data: str) -> Union[str, np.ndarray]:
        """Prepare input for local inference.

        Args:
            input_data: Input string, which can be a file path, URL, or Base64-encoded data.

        Returns:
            Either a file path/URL string or a numpy array for Base64-decoded images.

        Raises:
            ValueError: If the input format is invalid or Base64 decoding fails.
        """
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

    async def execute(
        self, input_data: str, file_type: Optional[str] = None, **options
    ) -> Dict[str, Any]:
        """Execute inference on the input data.

        Args:
            input_data: Input string (file path, URL, or Base64-encoded data).
            file_type: Unused parameter, kept for API compatibility. The file type
                       is inferred from the input data.
            **options: Additional options passed to the inference engine.

        Returns:
            A dictionary containing the inference results.

        Raises:
            RuntimeError: If the engine wrapper is not initialized.
            ValueError: If the input data format is invalid.
        """
        if not self._engine_wrapper:
            raise RuntimeError("Engine wrapper not initialized")

        processed_input = self._process_input_for_local(input_data)

        # Call inference
        result = await self._engine_wrapper.call(
            self._engine_wrapper.engine.predict, processed_input
        )

        return self._parse_result(result)

    def _parse_result(self, result: Any) -> Dict[str, Any]:
        """Parse local inference result into unified format.

        Args:
            result: The raw result from the inference engine.

        Returns:
            A dictionary containing the parsed result in a unified format.
        """
        if self._pipeline == "OCR":
            return self._parse_ocr_result(result)
        else:
            return self._parse_layout_result(result)

    def _parse_ocr_result(self, result: Any) -> Dict[str, Any]:
        """Parse OCR result"""
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
                    text_lines.append(
                        {
                            "text": text.strip(),
                            "confidence": round(conf, 3),
                            "bbox": boxes[i].tolist(),
                        }
                    )

        return {
            "text": "\n".join(clean_texts),
            "confidence": sum(confidences) / len(confidences) if confidences else 0,
            "text_lines": text_lines,
        }

    def _parse_layout_result(self, result: Any) -> Dict[str, Any]:
        """Parse layout parsing result"""
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
                    processed_images[img_key] = base64.b64encode(
                        buffer.getvalue()
                    ).decode("ascii")
            all_images_mapping.update(processed_images)

        return {
            "markdown": "\n".join(markdown_parts),
            "pages": len(result),
            "images_mapping": all_images_mapping,
        }
