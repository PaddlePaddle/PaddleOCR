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

import abc
from typing import Any, List, Optional, Union

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_context
from mcp.types import ImageContent, TextContent

from ..inference.base import Inference
from ..inference.types import InferenceRequest, InferenceResult

ToolReturn = Union[str, List[Union[TextContent, ImageContent]]]


class Task(abc.ABC):
    def __init__(self, inference: Inference):
        self._inference = inference

    @property
    @abc.abstractmethod
    def tool_name(self) -> str:
        pass

    @abc.abstractmethod
    def _format_result(
        self, result: InferenceResult, detailed: bool, **kwargs
    ) -> ToolReturn:
        pass

    def _tool_description(self) -> str:
        valid_params = sorted(self._inference.get_valid_params())
        params_hint = ", ".join(valid_params)
        return (
            f"PaddleOCR {self.tool_name} tool. "
            f"Optional runtime_params keys: {params_hint}."
        )

    async def _invoke_tool(
        self,
        input_data: str,
        output_mode: str = "simple",
        file_type: Optional[str] = None,
        return_images: bool = True,
        runtime_params: Optional[dict[str, Any]] = None,
    ) -> ToolReturn:
        ctx = get_context()
        await ctx.info(
            f"--- {self.tool_name} tool received input_data: {input_data[:50]} ---"
        )

        final_params = self._inference.get_final_params(runtime_params or {})
        self._inference.validate_params(final_params)

        request = InferenceRequest(
            input_data=input_data,
            file_type=file_type,
            runtime_params=final_params,
        )
        result = await self._inference.predict(request)

        return self._format_result(
            result,
            detailed=(output_mode == "detailed"),
            return_images=return_images,
        )

    def register_tools(self, mcp: FastMCP) -> None:
        mcp.tool(self.tool_name, description=self._tool_description())(
            self._invoke_tool
        )
