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

from .base import MCPCapability
from .ocr import OCRCapability
from .doc_parsing import DocParsingCapability


def create_capability(pipeline: str, executor) -> MCPCapability:
    """Create corresponding Capability based on pipeline type.

    Args:
        pipeline: Pipeline type.
        executor: Executor instance.

    Returns:
        MCPCapability instance.
    """
    if pipeline == "OCR":
        return OCRCapability(executor)
    elif pipeline == "PP-StructureV3":
        return DocParsingCapability(executor, tool_name="pp_structurev3")
    elif pipeline in ("PaddleOCR-VL", "PaddleOCR-VL-1.5", "PaddleOCR-VL-1.6"):
        return DocParsingCapability(executor, tool_name="paddleocr_vl")
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")
