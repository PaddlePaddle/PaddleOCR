# mcp_server/paddleocr_mcp/executors/qianfan.py
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

from typing import Any, Dict, Optional

from .http import HTTPExecutor


class QianfanExecutor(HTTPExecutor):
    """Executor for Qianfan platform HTTP API"""

    def __init__(self, base_url: str, api_key: str, pipeline: str, timeout: int = 60):
        super().__init__(base_url, timeout)
        self._api_key = api_key
        self._pipeline = pipeline

    def _get_headers(self) -> Dict[str, str]:
        """Return authentication headers for Qianfan API"""
        return {"Authorization": f"Bearer {self._api_key}"}

    def _get_endpoint(self) -> str:
        """Return the API endpoint for Qianfan"""
        return "paddleocr"

    def _prepare_payload(
        self, input_data: str, file_type: Optional[str], **options
    ) -> Dict[str, Any]:
        """Prepare request payload for Qianfan API"""
        payload = {"file": input_data}
        if file_type == "image":
            payload["fileType"] = 1
        elif file_type == "pdf":
            payload["fileType"] = 0
        return payload

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Qianfan API response into unified format"""
        result_data = response.get("result", response)

        if self._pipeline == "OCR":
            return self._parse_ocr_response_http(result_data)
        else:
            return self._parse_layout_response_http(result_data)
