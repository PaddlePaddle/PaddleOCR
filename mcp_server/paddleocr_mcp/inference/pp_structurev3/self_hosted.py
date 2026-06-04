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

from typing import Any
from typing_extensions import override

from ..shared.http_base import HTTPInferenceBase
from ..types import InferenceResult
from ..shared.http_result_parsers import parse_doc_parsing_result
from .params import PP_STRUCTUREV3_DEFAULT_PARAMS, PP_STRUCTUREV3_RUNTIME_PARAMS


class PPStructureV3SelfHostedInference(HTTPInferenceBase):
    def __init__(self, base_url: str, timeout: int = 60):
        super().__init__(base_url, timeout, api_key=None)

    @override
    def _get_endpoint(self) -> str:
        return "layout-parsing"

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
