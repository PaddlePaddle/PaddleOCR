# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ConvertResult:
    """Conversion result."""

    markdown: str
    title: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    images: dict = field(default_factory=dict)  # {relative_path: image_bytes}


class BaseConverter(ABC):
    """Abstract base class for all format converters."""

    supported_extensions: list[str] = []
    supported_mimetypes: list[str] = []

    @abstractmethod
    def convert_file(self, file_path: Path, **kwargs) -> ConvertResult:
        """Convert a file to Markdown."""
        ...
