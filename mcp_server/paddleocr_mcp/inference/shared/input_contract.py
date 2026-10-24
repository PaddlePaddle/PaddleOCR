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

"""Shared user-facing input contract for all MCP inference providers."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Generator, Optional

from ...utils import (
    decode_base64_payload,
    extract_base64_payload,
    infer_file_type_from_bytes,
    is_base64,
    is_url,
)

COMMON_INPUT_DESCRIPTION = (
    "Supported input_data: absolute file path accessible to the MCP server process "
    "(~ is expanded), HTTP(S) URL, raw Base64, or data URL. Relative paths are "
    "rejected. Backend-specific conversion is handled by the MCP server."
)

_INPUT_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".pdf",
        ".bmp",
        ".webp",
        ".tiff",
        ".tif",
        ".gif",
    }
)

_SUFFIX_BY_FILE_TYPE = {
    "image": ".png",
    "pdf": ".pdf",
}


class InputKind(str, Enum):
    URL = "url"
    ABSOLUTE_PATH = "absolute_path"
    BASE64 = "base64"
    DATA_URL = "data_url"


def normalize(input_data: str) -> str:
    return input_data.strip()


def looks_like_file_path(value: str) -> bool:
    if is_url(value) or value.startswith("data:"):
        return False
    if value.startswith("~") or "/" in value or "\\" in value:
        return True
    return Path(value).suffix.lower() in _INPUT_EXTENSIONS


def classify_input(value: str) -> InputKind:
    if is_url(value):
        return InputKind.URL
    if value.startswith("data:"):
        payload = extract_base64_payload(value)
        if not is_base64(payload):
            raise ValueError("Invalid data URL: payload is not valid Base64.")
        return InputKind.DATA_URL
    if is_base64(value):
        return InputKind.BASE64
    if looks_like_file_path(value):
        return InputKind.ABSOLUTE_PATH
    raise ValueError(f"Invalid input_data. {COMMON_INPUT_DESCRIPTION}")


def resolve_absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"Relative paths are not supported: {value!r}. "
            "Use an absolute path accessible to the MCP server process."
        )
    if not path.exists():
        raise ValueError(
            f"File not found: {value!r} (resolved to {path}). "
            f"{COMMON_INPUT_DESCRIPTION}"
        )
    return path.resolve()


def validate_external_contract(input_data: str) -> InputKind:
    value = normalize(input_data)
    if not value:
        raise ValueError("input_data must not be empty.")

    kind = classify_input(value)

    if kind is InputKind.ABSOLUTE_PATH:
        resolve_absolute_path(value)
        return kind

    if kind in (InputKind.BASE64, InputKind.DATA_URL):
        payload = extract_base64_payload(value)
        data = decode_base64_payload(payload)
        file_type = infer_file_type_from_bytes(data)
        if file_type not in ("image", "pdf"):
            raise ValueError(
                "Unsupported Base64 content type. Only image and PDF inputs are "
                "supported."
            )

    return kind


def decode_input_bytes(value: str, kind: InputKind) -> bytes:
    if kind is InputKind.ABSOLUTE_PATH:
        return resolve_absolute_path(value).read_bytes()
    payload = extract_base64_payload(value)
    return decode_base64_payload(payload)


@contextmanager
def materialize_bytes_as_temp_file(
    data: bytes,
    *,
    file_type: str,
) -> Generator[Path, None, None]:
    suffix = _SUFFIX_BY_FILE_TYPE.get(file_type, ".bin")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)
    try:
        temp_file.write(data)
        temp_file.close()
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)
