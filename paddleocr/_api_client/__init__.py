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

from .client import APIClient
from .async_client import AsyncAPIClient
from .models import (
    DocParsingOptions,
    Model,
    OCROptions,
    is_document_parsing_model,
    is_ocr_model,
)
from .results import (
    DocParsingPage,
    DocParsingResult,
    Job,
    JobStatus,
    OCRPage,
    OCRResult,
    Progress,
    ResourceSaveSummary,
)
from .errors import (
    APIError,
    AuthError,
    FileNotFoundError,
    InvalidRequestError,
    JobFailedError,
    NetworkError,
    PaddleOCRAPIError,
    PollTimeoutError,
    RequestTimeoutError,
    ResponseFormatError,
    ResultParseError,
)

__all__ = [
    "APIClient",
    "AsyncAPIClient",
    "Model",
    "OCROptions",
    "DocParsingOptions",
    "is_ocr_model",
    "is_document_parsing_model",
    "OCRResult",
    "OCRPage",
    "DocParsingResult",
    "DocParsingPage",
    "Job",
    "JobStatus",
    "Progress",
    "ResourceSaveSummary",
    "PaddleOCRAPIError",
    "AuthError",
    "InvalidRequestError",
    "APIError",
    "JobFailedError",
    "RequestTimeoutError",
    "PollTimeoutError",
    "FileNotFoundError",
    "ResponseFormatError",
    "ResultParseError",
    "NetworkError",
]
