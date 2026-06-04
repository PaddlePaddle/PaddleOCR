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
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

from ..base import Inference
from ..errors import InferenceError
from ..types import InferenceRequest, InferenceResult
from .async_http_client import AsyncHTTPClient
from .param_mapping import convert_params_to_camel


class HTTPInferenceBase(Inference):
    def __init__(
        self,
        base_url: str,
        timeout: int = 60,
        api_key: Optional[str] = None,
    ):
        self._base_url = base_url
        self._timeout = timeout
        self._api_key = api_key
        self._client: Optional[AsyncHTTPClient] = None

    async def start(self) -> None:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._client = AsyncHTTPClient(self._base_url, self._timeout, headers)
        await self._client.start()

    async def stop(self) -> None:
        if self._client:
            await self._client.stop()
            self._client = None

    async def predict(self, request: InferenceRequest) -> InferenceResult:
        if not self._client:
            raise RuntimeError("Inference not started")

        payload = self._prepare_payload(
            request.input_data,
            request.file_type,
            **request.runtime_params,
        )

        try:
            response = await self._client.post(self._get_endpoint(), payload)
            return self._parse_result(response)
        except Exception as e:
            raise InferenceError(f"HTTP request failed: {e}") from e

    def _encode_file_field(self, input_data: str) -> str:
        """Encode local paths to Base64; pass through URLs and raw payloads."""
        if input_data.startswith("http://") or input_data.startswith("https://"):
            return input_data
        from .local_input import LocalInputProcessor

        if LocalInputProcessor.is_file_path(input_data):
            path = Path(input_data).expanduser()
            return base64.b64encode(path.read_bytes()).decode("ascii")
        return input_data

    def _prepare_payload(
        self, input_data: str, file_type: Optional[str], **params
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"file": self._encode_file_field(input_data)}
        if file_type == "image":
            payload["fileType"] = 1
        elif file_type == "pdf":
            payload["fileType"] = 0

        payload.update(convert_params_to_camel({**params, "visualize": False}))
        return payload

    @abstractmethod
    def _get_endpoint(self) -> str:
        pass

    @abstractmethod
    def _get_params(self) -> tuple[set[str], dict[str, Any]]:
        pass

    @abstractmethod
    def _parse_result(self, response: dict[str, Any]) -> InferenceResult:
        pass

    def get_valid_params(self) -> set[str]:
        valid_params, _ = self._get_params()
        return valid_params

    def get_default_params(self) -> dict[str, Any]:
        _, defaults = self._get_params()
        return defaults
