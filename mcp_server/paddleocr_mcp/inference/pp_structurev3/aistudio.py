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
    PPStructureV3Options,
)

from ..base import Inference
from ..errors import (
    AuthenticationError,
    ExecutionTimeoutError,
    InferenceError,
    ResourceUnavailableError,
)
from ..shared.doc_parsing_result_adapters import parse_aistudio_doc_parsing_result
from ..types import DocParsingResult, InferenceRequest
from .params import PP_STRUCTUREV3_DEFAULT_PARAMS, PP_STRUCTUREV3_RUNTIME_PARAMS


class PPStructureV3AIStudioInference(Inference):
    def __init__(
        self,
        token: str,
        base_url: Optional[str] = None,
        request_timeout: float = 300.0,
        poll_timeout: float = 600.0,
    ):
        self._token = token
        self._base_url = base_url
        self._request_timeout = request_timeout
        self._poll_timeout = poll_timeout
        self._client = None

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
            input_source = self._resolve_input_source(request.input_data)
            options = PPStructureV3Options(**request.runtime_params, visualize=False)

            result = await self._client.parse_document(
                model=Model.PP_STRUCTURE_V3, **input_source, options=options
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
        return set(PP_STRUCTUREV3_RUNTIME_PARAMS.keys())

    def get_default_params(self) -> dict[str, object]:
        return PP_STRUCTUREV3_DEFAULT_PARAMS.copy()

    def _resolve_input_source(self, input_data: str) -> dict[str, str]:
        if input_data.startswith("http://") or input_data.startswith("https://"):
            return {"file_url": input_data}
        return {"file_path": input_data}

    def _parse_result(self, result) -> DocParsingResult:
        return parse_aistudio_doc_parsing_result(result)
