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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class OCRPage:
    pruned_result: Any
    ocr_image_url: Optional[str] = None


@dataclass
class DocParsingPage:
    markdown_text: str
    markdown_images: Dict[str, str] = field(default_factory=dict)
    output_images: Dict[str, str] = field(default_factory=dict)


@dataclass
class OCRResult:
    job_id: str
    pages: List[OCRPage] = field(default_factory=list)


@dataclass
class DocParsingResult:
    job_id: str
    pages: List[DocParsingPage] = field(default_factory=list)


@dataclass
class Progress:
    total_pages: int = 0
    extracted_pages: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@dataclass
class Job:
    job_id: str
    model: str
    task: Literal["ocr", "document_parsing"]


@dataclass
class JobStatus:
    job_id: str
    state: str
    progress: Optional[Progress] = None
    result: Any = None
    error_msg: Optional[str] = None


@dataclass
class BatchStatus:
    batch_id: str
    jobs: List[JobStatus] = field(default_factory=list)
