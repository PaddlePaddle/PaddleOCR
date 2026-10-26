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

"""Tests for MCP input classification of file paths vs. Base64 (issue #18412)."""

import base64
import io
import os

import pytest

pytest.importorskip("fastmcp")
pytest.importorskip("puremagic")

from paddleocr_mcp.inference.shared.input_contract import (  # noqa: E402
    InputKind,
    classify_input,
    validate_external_contract,
)
from paddleocr_mcp.utils import is_base64  # noqa: E402

pytestmark = pytest.mark.py38_incompatible


def _png_base64() -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize(
    "value, expected",
    [
        ("aGVsbG8=", True),
        ("aGVsbG8gd29ybGQ=", True),
        ("/9j/4AAQ", True),  # JPEG prefix starts with "/"
        ("aGVsbG8", False),  # length not a multiple of 4
        ("/etc/passwd", False),  # length 11
        ("aGVs=G8=", False),  # padding in the middle
        ("aGVsbG8*", False),  # character outside the alphabet
        ("", False),
    ],
)
def test_is_base64(value, expected):
    assert is_base64(value) is expected


# Existing absolute paths made only of Base64-alphabet characters.
# "/usr/bin/env" is 12 characters long, so it is also decodable Base64.
@pytest.mark.parametrize("path", ["/etc/passwd", "/usr/bin/env", "/bin/sh"])
def test_existing_alphabet_only_path_is_classified_as_path(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} does not exist on this platform")
    assert classify_input(path) is InputKind.ABSOLUTE_PATH
    assert validate_external_contract(path) is InputKind.ABSOLUTE_PATH


def test_raw_base64_is_still_classified_as_base64():
    value = _png_base64()
    assert classify_input(value) is InputKind.BASE64
    assert validate_external_contract(value) is InputKind.BASE64


def test_base64_starting_with_slash_is_still_base64():
    # JPEG data encodes to a string starting with "/9j/", which also looks
    # like an absolute path; it must not be treated as one unless it exists.
    value = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 32).decode()
    assert value.startswith("/9j/")
    assert classify_input(value) is InputKind.BASE64


def test_missing_alphabet_only_path_error_mentions_path():
    # 32 Base64-alphabet characters: decodable, but not an image or PDF.
    missing = "/nonexistent0dir/uploads/receipt"
    assert not os.path.exists(missing)
    with pytest.raises(ValueError, match="the file does not exist"):
        validate_external_contract(missing)
