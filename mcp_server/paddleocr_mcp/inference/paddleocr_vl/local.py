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

from ..base import Inference
from ..shared.doc_parsing_result_adapters import parse_local_doc_parsing_result
from ..shared.input_adapters import LOCAL_INPUT_ADAPTER, InputAdapter
from ..shared.local_sync_runner import LocalSyncRunner
from ..types import DocParsingResult, InferenceRequest
from .params import PADDLEOCR_VL_DEFAULT_PARAMS, PADDLEOCR_VL_RUNTIME_PARAMS

try:
    from paddleocr import PaddleOCRVL

    LOCAL_VL_AVAILABLE = True
except ImportError:
    LOCAL_VL_AVAILABLE = False

_PIPELINE_VERSION_BY_MODEL = {
    "PaddleOCR-VL": "v1",
    "PaddleOCR-VL-1.5": "v1.5",
    "PaddleOCR-VL-1.6": "v1.6",
}


class PaddleOCRVLLocalInference(Inference):
    def __init__(
        self,
        config: Optional[str] = None,
        device: Optional[str] = None,
        model: str = "PaddleOCR-VL",
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
        if not LOCAL_VL_AVAILABLE:
            raise RuntimeError("PaddleOCRVL is not locally available")
        try:
            pipeline_version = _PIPELINE_VERSION_BY_MODEL[self._model]
            self._inference = PaddleOCRVL(
                pipeline_version=pipeline_version,
                paddlex_config=self._config,
                device=self._device,
            )
            self._wrapper = LocalSyncRunner(self._inference)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create PaddleOCRVL inference: {str(e)}"
            ) from e

    async def stop(self) -> None:
        if self._wrapper:
            await self._wrapper.close()
            self._wrapper = None

    async def predict(self, request: InferenceRequest) -> DocParsingResult:
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
        return set(PADDLEOCR_VL_RUNTIME_PARAMS.keys())

    def get_default_params(self) -> dict[str, Any]:
        return PADDLEOCR_VL_DEFAULT_PARAMS.copy()

    def _parse_result(self, result: Any) -> DocParsingResult:
        return parse_local_doc_parsing_result(result)
