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

"""MCP ImageContent formatting helpers for tool output."""

from __future__ import annotations

import base64
from typing import Optional, Tuple

import httpx

from ..utils import (
    decode_base64_payload,
    extract_base64_payload,
    infer_file_type_from_bytes,
    is_url,
)


def _mime_type_from_image_bytes(data: bytes) -> str:
    if infer_file_type_from_bytes(data) != "image":
        return "image/jpeg"
    import puremagic

    mime = puremagic.from_string(data, mime=True)
    if mime.startswith("image/"):
        return mime
    return "image/jpeg"


def normalize_mcp_image_payload(raw: str) -> Optional[Tuple[str, str]]:
    """Convert provider image payloads to MCP ImageContent (base64 + mimeType)."""
    value = raw.strip()
    if not value:
        return None

    if is_url(value):
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(value)
                response.raise_for_status()
            data = response.content
            return (
                base64.b64encode(data).decode("ascii"),
                _mime_type_from_image_bytes(data),
            )
        except Exception:
            return None

    if value.startswith("data:"):
        try:
            data = decode_base64_payload(extract_base64_payload(value))
            mime = "image/jpeg"
            header = value[5:].split(",", 1)[0]
            if ";" in header:
                declared = header.split(";", 1)[0]
                if declared.startswith("image/"):
                    mime = declared
            if mime == "image/jpeg":
                mime = _mime_type_from_image_bytes(data)
            return base64.b64encode(data).decode("ascii"), mime
        except ValueError:
            return None

    try:
        data = decode_base64_payload(value)
        return value, _mime_type_from_image_bytes(data)
    except ValueError:
        return None
