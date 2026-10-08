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

import pytest

from paddleocr_mcp.providers import (
    InferenceProvider,
    ProviderTransport,
    get_provider_spec,
    http_providers,
    is_http_provider,
    normalize_provider,
    provider_choices,
)


class TestNormalizeProvider:
    def test_string_is_converted_to_enum(self) -> None:
        assert normalize_provider("qianfan") is InferenceProvider.QIANFAN

    def test_enum_is_returned_as_is(self) -> None:
        assert normalize_provider(InferenceProvider.LOCAL) is InferenceProvider.LOCAL

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_provider("not-a-provider")


class TestProviderSpecs:
    def test_every_provider_has_a_spec(self) -> None:
        for provider in InferenceProvider:
            assert get_provider_spec(provider).provider is provider

    def test_transports_are_mapped(self) -> None:
        assert get_provider_spec("local").transport is ProviderTransport.LOCAL
        assert (
            get_provider_spec("aistudio").transport
            is ProviderTransport.AISTUDIO_API
        )
        assert get_provider_spec("qianfan").transport is ProviderTransport.HTTP


class TestHttpProviders:
    def test_qianfan_and_self_hosted_are_http(self) -> None:
        assert is_http_provider("qianfan")
        assert is_http_provider("self_hosted")

    def test_local_is_not_http(self) -> None:
        assert not is_http_provider("local")

    def test_http_providers_set(self) -> None:
        assert http_providers() == {
            InferenceProvider.QIANFAN,
            InferenceProvider.SELF_HOSTED,
        }


class TestProviderChoices:
    def test_choices_match_enum_values(self) -> None:
        assert provider_choices() == [provider.value for provider in InferenceProvider]

    def test_choices_are_unique(self) -> None:
        choices = provider_choices()
        assert len(choices) == len(set(choices))
