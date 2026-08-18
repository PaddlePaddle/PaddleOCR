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

from dataclasses import dataclass
from enum import Enum


class InferenceProvider(str, Enum):
    LOCAL = "local"
    AISTUDIO = "aistudio"
    QIANFAN = "qianfan"
    SELF_HOSTED = "self_hosted"


class ProviderTransport(str, Enum):
    LOCAL = "local"
    AISTUDIO_API = "aistudio_api"
    HTTP = "http"


@dataclass(frozen=True)
class ProviderSpec:
    provider: InferenceProvider
    transport: ProviderTransport
    display_name: str


_PROVIDER_SPECS: dict[InferenceProvider, ProviderSpec] = {
    InferenceProvider.LOCAL: ProviderSpec(
        provider=InferenceProvider.LOCAL,
        transport=ProviderTransport.LOCAL,
        display_name="local",
    ),
    InferenceProvider.AISTUDIO: ProviderSpec(
        provider=InferenceProvider.AISTUDIO,
        transport=ProviderTransport.AISTUDIO_API,
        display_name="AI Studio",
    ),
    InferenceProvider.QIANFAN: ProviderSpec(
        provider=InferenceProvider.QIANFAN,
        transport=ProviderTransport.HTTP,
        display_name="Qianfan",
    ),
    InferenceProvider.SELF_HOSTED: ProviderSpec(
        provider=InferenceProvider.SELF_HOSTED,
        transport=ProviderTransport.HTTP,
        display_name="self-hosted",
    ),
}


def normalize_provider(provider: InferenceProvider | str) -> InferenceProvider:
    if isinstance(provider, InferenceProvider):
        return provider
    return InferenceProvider(provider)


def get_provider_spec(provider: InferenceProvider | str) -> ProviderSpec:
    return _PROVIDER_SPECS[normalize_provider(provider)]


def provider_choices() -> list[str]:
    return [provider.value for provider in InferenceProvider]


def http_providers() -> set[InferenceProvider]:
    return {
        spec.provider
        for spec in _PROVIDER_SPECS.values()
        if spec.transport is ProviderTransport.HTTP
    }


def is_http_provider(provider: InferenceProvider | str) -> bool:
    return get_provider_spec(provider).transport is ProviderTransport.HTTP
