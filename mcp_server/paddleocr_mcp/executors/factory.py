# mcp_server/paddleocr_mcp/executors/factory.py
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

from .aistudio import AIStudioExecutor
from .base import Executor
from .local import LocalExecutor
from .qianfan import QianfanExecutor
from .self_hosted import SelfHostedExecutor


def create_executor(
    source: str,
    pipeline: str,
    token: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 60,
    api_key: Optional[str] = None,
    pipeline_config: Optional[str] = None,
    device: Optional[str] = None,
) -> Executor:
    """Create an Executor instance based on the specified source.

    Args:
        source: Executor source - "local", "aistudio", "qianfan", or "self_hosted"
        pipeline: Pipeline type - "OCR", "PP-StructureV3", "PaddleOCR-VL",
            "PaddleOCR-VL-1.5", or "PaddleOCR-VL-1.6"
        token: AI Studio access token (required for aistudio source)
        base_url: Service base URL (required for qianfan/self_hosted, optional for aistudio)
        timeout: Timeout in seconds (default: 60)
        api_key: Qianfan API key (required for qianfan source)
        pipeline_config: Pipeline config file path (for local source)
        device: Device for inference (for local source)

    Returns:
        Executor instance

    Raises:
        ValueError: If required parameters are missing or unknown source
    """
    if source == "local":
        return LocalExecutor(
            pipeline=pipeline,
            pipeline_config=pipeline_config,
            device=device,
        )
    elif source == "aistudio":
        return AIStudioExecutor(
            pipeline=pipeline,
            token=token,
            base_url=base_url,
            request_timeout=float(timeout),
            poll_timeout=float(timeout * 10),
        )
    elif source == "qianfan":
        if not base_url:
            raise ValueError("base_url is required for qianfan source")
        if not api_key:
            raise ValueError("api_key is required for qianfan source")
        return QianfanExecutor(
            base_url=base_url,
            api_key=api_key,
            pipeline=pipeline,
            timeout=timeout,
        )
    elif source == "self_hosted":
        if not base_url:
            raise ValueError("base_url is required for self_hosted source")
        return SelfHostedExecutor(
            base_url=base_url,
            pipeline=pipeline,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown source: {source}")
