# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Any, Optional

from paddleocr import PaddleOCR

from ..base import Inference
from ..shared.input_adapters import LOCAL_INPUT_ADAPTER, InputAdapter
from ..shared.local_sync_runner import LocalSyncRunner
from ..types import InferenceRequest, OCRResult, TextLine
from .params import OCR_DEFAULT_PARAMS, OCR_RUNTIME_PARAMS

_LOCAL_OCR_INIT_BY_MODEL: dict[str, dict[str, str]] = {
    "PP-OCRv5": {"ocr_version": "PP-OCRv5"},
    "PP-OCRv6": {"ocr_version": "PP-OCRv6"},
    "PP-OCRv5-latin": {
        "text_detection_model_name": "PP-OCRv5_server_det",
        "text_recognition_model_name": "latin_PP-OCRv5_mobile_rec",
    },
}


def _build_local_ocr_init_kwargs(
    *,
    config: Optional[str],
    device: Optional[str],
    model: Optional[str],
) -> dict[str, Any]:
    init_kwargs: dict[str, Any] = {
        "paddlex_config": config,
        "device": device,
    }
    if config is None and model is not None:
        model_kwargs = _LOCAL_OCR_INIT_BY_MODEL.get(model)
        if model_kwargs is None:
            raise ValueError(f"Unsupported local OCR model: {model!r}")
        init_kwargs.update(model_kwargs)
    return init_kwargs


class OCRLocalInference(Inference):
    def __init__(
        self,
        config: Optional[str] = None,
        device: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self._config = config
        self._device = device
        self._model = model
        self._inference: Optional[Any] = None
        self._wrapper: Optional[LocalSyncRunner] = None

    @property
    def input_adapter(self) -> InputAdapter:
        return LOCAL_INPUT_ADAPTER

    async def start(self) -> None:
        try:
            init_kwargs = _build_local_ocr_init_kwargs(
                config=self._config,
                device=self._device,
                model=self._model,
            )
            self._inference = PaddleOCR(**init_kwargs)
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

        with self.input_adapter.prepare(request.input_data) as processed_input:
            result = await self._wrapper.call(
                self._wrapper.inference.predict,
                processed_input,
                **request.runtime_params,
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
