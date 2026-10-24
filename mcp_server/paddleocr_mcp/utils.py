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

"""Encoding helpers shared by input contract and inference adapters."""

from __future__ import annotations

import base64
import re
from typing import Optional
from urllib.parse import urlparse


def is_url(value: str) -> bool:
    if not (value.startswith("http://") or value.startswith("https://")):
        return False

    result = urlparse(value)
    return all([result.scheme, result.netloc]) and result.scheme in ("http", "https")


def is_base64(value: str) -> bool:
    pattern = r"^[A-Za-z0-9+/]+={0,2}$"
    return bool(re.fullmatch(pattern, value))


def extract_base64_payload(input_data: str) -> str:
    if input_data.startswith("data:"):
        if "," not in input_data:
            raise ValueError("Invalid data URL: expected a comma after the MIME type.")
        return input_data.split(",", 1)[1]
    return input_data


def decode_base64_payload(payload: str) -> bytes:
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as e:
        raise ValueError(
            f"Invalid Base64 input: {e}. "
            "Ensure the string is complete and correctly padded."
        ) from e


def infer_file_type_from_bytes(data: bytes) -> Optional[str]:
    import puremagic

    mime = puremagic.from_string(data, mime=True)
    if mime.startswith("image/"):
        return "image"
    if mime == "application/pdf":
        return "pdf"
    return None
