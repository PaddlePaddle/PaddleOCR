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

from __future__ import annotations

import base64

import pytest

from paddleocr_mcp.utils import (
    decode_base64_payload,
    extract_base64_payload,
    is_base64,
    is_url,
)


class TestIsUrl:
    def test_http_and_https_are_urls(self) -> None:
        assert is_url("http://example.com")
        assert is_url("https://example.com/doc.pdf")

    def test_missing_scheme_is_not_url(self) -> None:
        assert not is_url("example.com")
        assert not is_url("/tmp/doc.pdf")

    def test_scheme_without_host_is_not_url(self) -> None:
        assert not is_url("http://")

    def test_non_http_scheme_is_not_url(self) -> None:
        assert not is_url("ftp://example.com")


class TestIsBase64:
    def test_padded_base64_is_detected(self) -> None:
        assert is_base64("aGVsbG8=")

    def test_empty_string_is_not_base64(self) -> None:
        assert not is_base64("")

    def test_strings_with_illegal_characters_are_not_base64(self) -> None:
        assert not is_base64("not base64!!")

    def test_padding_only_is_not_base64(self) -> None:
        assert not is_base64("====")


class TestExtractBase64Payload:
    def test_data_url_returns_payload_after_comma(self) -> None:
        assert (
            extract_base64_payload("data:image/png;base64,AAAA") == "AAAA"
        )

    def test_plain_string_is_returned_unchanged(self) -> None:
        assert extract_base64_payload("AAAA") == "AAAA"

    def test_data_url_without_comma_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_base64_payload("data:image/png;base64")


class TestDecodeBase64Payload:
    def test_valid_payload_is_decoded(self) -> None:
        encoded = base64.b64encode(b"hello").decode()
        assert decode_base64_payload(encoded) == b"hello"

    def test_invalid_payload_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            decode_base64_payload("not base64!!")

    def test_missing_padding_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            decode_base64_payload("aGVsbG8")
