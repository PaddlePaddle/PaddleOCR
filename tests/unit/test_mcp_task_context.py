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

"""Tests for calling MCP tools without an MCP request context (issue #18391)."""

import asyncio
import base64
import io

import pytest

pytest.importorskip("fastmcp")
pytest.importorskip("puremagic")

from fastmcp import Client, FastMCP  # noqa: E402
from fastmcp.server.context import Context  # noqa: E402

from paddleocr_mcp.inference.base import Inference  # noqa: E402
from paddleocr_mcp.inference.shared.input_adapters import (  # noqa: E402
    LocalInputAdapter,
)
from paddleocr_mcp.inference.types import DocParsingResult  # noqa: E402
from paddleocr_mcp.tasks.doc_parsing import DocParsingTask  # noqa: E402

pytestmark = pytest.mark.py38_incompatible


class _StubInference(Inference):
    @property
    def input_adapter(self):
        return LocalInputAdapter()

    async def start(self):
        pass

    async def stop(self):
        pass

    async def predict(self, request):
        return DocParsingResult(markdown="# Parsed", pages=1, images_mapping={})

    def get_valid_params(self):
        return set()

    def get_default_params(self):
        return {}


class _PPStructureV3Task(DocParsingTask):
    @property
    def tool_name(self):
        return "pp_structurev3"


def _png_base64() -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_server():
    task = _PPStructureV3Task(_StubInference())
    mcp = FastMCP("test", mask_error_details=False)
    task.register_tools(mcp)
    return task, mcp


def test_tool_without_any_context():
    task, _ = _make_server()
    result = asyncio.run(task._invoke_tool(_png_base64(), return_images=False))
    assert result.text == "# Parsed"


def test_tool_with_context_but_no_request_context():
    task, mcp = _make_server()

    async def run():
        async with Context(fastmcp=mcp) as ctx:
            assert ctx.request_context is None
            return await task._invoke_tool(_png_base64(), return_images=False)

    result = asyncio.run(run())
    assert result.text == "# Parsed"


def test_tool_through_mcp_client_still_notifies():
    _, mcp = _make_server()
    messages = []

    async def log_handler(message):
        messages.append(message)

    async def run():
        async with Client(mcp, log_handler=log_handler) as client:
            return await client.call_tool(
                "pp_structurev3",
                {"input_data": _png_base64(), "return_images": False},
            )

    result = asyncio.run(run())
    assert result.content[0].text == "# Parsed"
    assert any("pp_structurev3 tool received input_data" in str(m) for m in messages)
