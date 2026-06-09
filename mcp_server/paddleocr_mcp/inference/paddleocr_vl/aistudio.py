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

from typing import Optional

from ..shared.paddleocr_api_sdk import (
    AsyncPaddleOCRClient,
    APIError,
    AuthError,
    JobFailedError,
    RequestTimeoutError,
    ResponseFormatError,
    ResultParseError,
    ServiceUnavailableError,
    Model,
    PaddleOCRVLOptions,
    resolve_document_model,
)

from ..base import Inference
from ..errors import (
    AuthenticationError,
    ExecutionTimeoutError,
    InferenceError,
    ResourceUnavailableError,
)
from ..shared.doc_parsing_result_adapters import parse_aistudio_doc_parsing_result
from ..shared.input_adapters import AISTUDIO_INPUT_ADAPTER, InputAdapter
from ..types import DocParsingResult, InferenceRequest
from .params import PADDLEOCR_VL_DEFAULT_PARAMS, PADDLEOCR_VL_RUNTIME_PARAMS


class PaddleOCRVLAIStudioInference(Inference):
    def __init__(
        self,
        token: str,
        base_url: Optional[str] = None,
        request_timeout: float = 120.0,
        poll_timeout: float = 600.0,
        model: str = Model.PADDLE_OCR_VL.value,
    ):
        self._token = token
        self._base_url = base_url
        self._request_timeout = request_timeout
        self._poll_timeout = poll_timeout
        self._model = resolve_document_model(model)
        self._client = None

    @property
    def input_adapter(self) -> InputAdapter:
        return AISTUDIO_INPUT_ADAPTER

    async def start(self) -> None:
        try:
            self._client = AsyncPaddleOCRClient(
                token=self._token,
                base_url=self._base_url,
                request_timeout=self._request_timeout,
                poll_timeout=self._poll_timeout,
            )
        except AuthError as e:
            raise AuthenticationError(f"Authentication failed: {e}")

    async def stop(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def predict(self, request: InferenceRequest) -> DocParsingResult:
        if not self._client:
            raise RuntimeError("Inference not started")

        try:
            with self.input_adapter.prepare(request.input_data) as input_payload:
                options = PaddleOCRVLOptions(**request.runtime_params)

                model = self._model
                result = await self._client.parse_document(
                    model=model, **input_payload, options=options
                )

            return self._parse_result(result)

        except AuthError as e:
            raise AuthenticationError(f"Authentication failed: {e}")
        except ServiceUnavailableError as e:
            raise ResourceUnavailableError(f"Service unavailable: {e}")
        except (JobFailedError, APIError, ResponseFormatError, ResultParseError) as e:
            raise InferenceError(f"Execution failed: {e}")
        except RequestTimeoutError as e:
            raise ExecutionTimeoutError(f"Request timeout: {e}")

    def get_valid_params(self) -> set[str]:
        return set(PADDLEOCR_VL_RUNTIME_PARAMS.keys())

    def get_default_params(self) -> dict[str, object]:
        return PADDLEOCR_VL_DEFAULT_PARAMS.copy()

    def _parse_result(self, result) -> DocParsingResult:
        return parse_aistudio_doc_parsing_result(result)
