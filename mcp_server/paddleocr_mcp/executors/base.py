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

import abc
from typing import Any, Dict, Optional


class ExecutorError(RuntimeError):
    """Base class for executor errors."""


class AuthenticationError(ExecutorError):
    """Authentication failed."""


class ResourceUnavailableError(ExecutorError):
    """Service unavailable."""


class ExecutionTimeoutError(ExecutorError):
    """Request timeout."""


class Executor(abc.ABC):
    """Abstract base class for executors that handle underlying inference execution."""

    @abc.abstractmethod
    async def execute(
        self, input_data: str, file_type: Optional[str] = None, **options
    ) -> Dict[str, Any]:
        """Execute inference and return unified result.

        Args:
            input_data: Input data (file path, URL, or base64).
            file_type: File type ("image" or "pdf").
            **options: Additional options.

        Returns:
            Unified result dictionary.
        """

    @abc.abstractmethod
    async def start(self) -> None:
        """Initialize resources."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Clean up resources."""
