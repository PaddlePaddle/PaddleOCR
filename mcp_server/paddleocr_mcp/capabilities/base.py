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
from typing import Any, Dict, List, Optional, Union

from fastmcp import Context
from mcp.types import ImageContent, TextContent

from ..executors.base import Executor


OutputMode = str  # "simple" or "detailed"


class MCPCapability(abc.ABC):
    """Abstract base class for MCP capabilities, responsible for tool registration and result formatting."""

    # Subclasses should define their pipeline
    PIPELINE: str

    # Default options for all pipelines (constant, read-only)
    DEFAULT_OPTIONS = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
    }

    def __init__(self, executor: Executor):
        self._executor = executor

    async def start(self) -> None:
        """Start the capability."""
        await self._executor.start()

    async def stop(self) -> None:
        """Stop the capability."""
        await self._executor.stop()

    def get_default_options(self) -> Dict[str, Any]:
        """Get default options for this capability.

        Returns:
            Copy of default options dictionary.
        """
        return self.DEFAULT_OPTIONS.copy()

    @abc.abstractmethod
    def register_tools(self, mcp: Any) -> None:
        """Register MCP tools.

        Args:
            mcp: FastMCP instance.
        """

    @abc.abstractmethod
    def _format_result(
        self, result: Dict[str, Any], detailed: bool, **kwargs
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        """Format result to MCP output.

        Args:
            result: Unified format result.
            detailed: Whether to use detailed format.
            **kwargs: Additional options.

        Returns:
            MCP output.
        """

    async def _process(
        self,
        input_data: str,
        output_mode: str,
        ctx: Context,
        file_type: Optional[str] = None,
        **options,
    ) -> Union[str, List[Union[TextContent, ImageContent]]]:
        """Process input and format output.

        Args:
            input_data: Input data.
            output_mode: Output mode.
            ctx: MCP context.
            file_type: File type.
            **options: Additional options.

        Returns:
            Formatted output.
        """
        # Get defaults from subclass method
        _defaults = self.get_default_options()
        # User options override defaults
        _defaults.update(options)

        result = await self._executor.execute(input_data, file_type, **_defaults)
        return self._format_result(result, output_mode == "detailed", **options)
