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

from typing import Any
from typing_extensions import override

from ...providers import InferenceProvider
from ..shared.http_base import HTTPInferenceBase
from ..types import InferenceResult
from ..shared.http_result_parsers import parse_doc_parsing_result
from .params import PP_STRUCTUREV3_DEFAULT_PARAMS, PP_STRUCTUREV3_RUNTIME_PARAMS


class PPStructureV3QianfanInference(HTTPInferenceBase):
    def __init__(self, base_url: str, api_key: str, http_timeout: int = 600):
        super().__init__(
            base_url, http_timeout, api_key, provider=InferenceProvider.QIANFAN
        )

    @override
    def _get_endpoint(self) -> str:
        return "paddleocr"

    @override
    def _get_params(self) -> tuple[set[str], dict[str, Any]]:
        return (
            set(PP_STRUCTUREV3_RUNTIME_PARAMS.keys()),
            PP_STRUCTUREV3_DEFAULT_PARAMS.copy(),
        )

    @override
    def _parse_result(self, response: dict[str, Any]) -> InferenceResult:
        result_data = response.get("result", response)
        return parse_doc_parsing_result(result_data)
