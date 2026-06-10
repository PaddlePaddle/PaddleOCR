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
import io
from typing import Any, Optional

from paddleocr import PaddleOCR

from ..base import Inference
from ..shared.local_sync_runner import LocalSyncRunner
from ..shared.local_input import LocalInputProcessor
from ..types import InferenceRequest, OCRResult, TextLine
from .params import OCR_DEFAULT_PARAMS, OCR_RUNTIME_PARAMS


class OCRLocalInference(Inference):
    def __init__(self, config: Optional[str] = None, device: Optional[str] = None):
        self._config = config
        self._device = device
        self._inference: Optional[Any] = None
        self._wrapper: Optional[LocalSyncRunner] = None

    async def start(self) -> None:
        try:
            self._inference = PaddleOCR(
                paddlex_config=self._config, device=self._device
            )
            self._wrapper = LocalSyncRunner(self._inference)
        except Exception as e:
            raise RuntimeError(f"Failed to create PaddleOCR inference: {str(e)}") from e

    async def stop(self) -> None:
        if self._wrapper:
            await self._wrapper.close()
            self._wrapper = None

    async def predict(self, request: InferenceRequest) -> OCRResult:
        if not self._wrapper:
            raise RuntimeError("Inference not started")

        processed_input = LocalInputProcessor.process_for_local(request.input_data)

        result = await self._wrapper.call(
            self._wrapper.inference.predict, processed_input, **request.runtime_params
        )

        return self._parse_result(result)

    def get_valid_params(self) -> set[str]:
        return set(OCR_RUNTIME_PARAMS.keys())

    def get_default_params(self) -> dict[str, Any]:
        return OCR_DEFAULT_PARAMS.copy()

    def _parse_result(self, result: Any) -> OCRResult:
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
                        TextLine(
                            text=text.strip(),
                            confidence=round(conf, 3),
                            bbox=boxes[i].tolist(),
                        )
                    )

        return OCRResult(
            text="\n".join(clean_texts),
            confidence=sum(confidences) / len(confidences) if confidences else 0,
            text_lines=text_lines,
        )
