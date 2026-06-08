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

from typing import Any, Callable

from .base import Inference
from .ocr.aistudio import OCRAIStudioInference
from .ocr.local import OCRLocalInference
from .ocr.self_hosted import OCRSelfHostedInference
from .paddleocr_vl.aistudio import PaddleOCRVLAIStudioInference
from .paddleocr_vl.local import PaddleOCRVLLocalInference
from .paddleocr_vl.qianfan import PaddleOCRVLQianfanInference
from .paddleocr_vl.self_hosted import PaddleOCRVLSelfHostedInference
from .pp_structurev3.aistudio import PPStructureV3AIStudioInference
from .pp_structurev3.local import PPStructureV3LocalInference
from .pp_structurev3.qianfan import PPStructureV3QianfanInference
from .pp_structurev3.self_hosted import PPStructureV3SelfHostedInference
from ..selection import ResolvedModel


class InferenceFactory:
    _registry: dict[tuple[str, str], Callable[..., Inference]] = {}

    @classmethod
    def register(
        cls,
        tool: str,
        source: str,
        factory_fn: Callable[..., Inference],
    ) -> None:
        cls._registry[(tool, source)] = factory_fn

    @classmethod
    def create(
        cls,
        resolved: ResolvedModel,
        source: str,
        **kwargs,
    ) -> Inference:
        key = (resolved.tool, source)
        if key not in cls._registry:
            raise ValueError(
                f"Unsupported inference combination: model={resolved.model!r}, "
                f"source={source!r}. Supported combinations: "
                f"{sorted(cls._registry.keys())}"
            )
        factory_fn = cls._registry[key]
        return factory_fn(resolved=resolved, **kwargs)

    @classmethod
    def list_supported(cls) -> set[tuple[str, str]]:
        return set(cls._registry.keys())


def _create_ocr_local(resolved: ResolvedModel, **kwargs: Any) -> Inference:
    return OCRLocalInference(
        config=kwargs.get("config"),
        device=kwargs.get("device"),
        ocr_version=resolved.ocr_version,
    )


def _create_ocr_aistudio(resolved: ResolvedModel, **kwargs: Any) -> Inference:
    return OCRAIStudioInference(
        token=kwargs["token"],
        base_url=kwargs.get("base_url"),
        request_timeout=kwargs.get("request_timeout", 300.0),
        poll_timeout=kwargs.get("poll_timeout", 600.0),
        model=resolved.model,
    )


def _create_ocr_self_hosted(resolved: ResolvedModel, **kwargs: Any) -> Inference:
    return OCRSelfHostedInference(
        base_url=kwargs["base_url"],
        timeout=kwargs.get("timeout", 60),
    )


def _create_pp_structurev3_local(resolved: ResolvedModel, **kwargs: Any) -> Inference:
    return PPStructureV3LocalInference(
        config=kwargs.get("config"),
        device=kwargs.get("device"),
    )


def _create_pp_structurev3_aistudio(
    resolved: ResolvedModel, **kwargs: Any
) -> Inference:
    return PPStructureV3AIStudioInference(
        token=kwargs["token"],
        base_url=kwargs.get("base_url"),
        request_timeout=kwargs.get("request_timeout", 300.0),
        poll_timeout=kwargs.get("poll_timeout", 600.0),
        model=resolved.model,
    )


def _create_pp_structurev3_self_hosted(
    resolved: ResolvedModel, **kwargs: Any
) -> Inference:
    return PPStructureV3SelfHostedInference(
        base_url=kwargs["base_url"],
        timeout=kwargs.get("timeout", 60),
    )


def _create_paddleocr_vl_local(resolved: ResolvedModel, **kwargs: Any) -> Inference:
    return PaddleOCRVLLocalInference(
        config=kwargs.get("config"),
        device=kwargs.get("device"),
        version=resolved.vl_version or "v1",
    )


def _create_paddleocr_vl_aistudio(resolved: ResolvedModel, **kwargs: Any) -> Inference:
    return PaddleOCRVLAIStudioInference(
        token=kwargs["token"],
        base_url=kwargs.get("base_url"),
        request_timeout=kwargs.get("request_timeout", 300.0),
        poll_timeout=kwargs.get("poll_timeout", 600.0),
        model=resolved.model,
    )


def _create_paddleocr_vl_self_hosted(
    resolved: ResolvedModel, **kwargs: Any
) -> Inference:
    return PaddleOCRVLSelfHostedInference(
        base_url=kwargs["base_url"],
        timeout=kwargs.get("timeout", 60),
    )


def _create_pp_structurev3_qianfan(resolved: ResolvedModel, **kwargs: Any) -> Inference:
    return PPStructureV3QianfanInference(
        base_url=kwargs["base_url"],
        api_key=kwargs["api_key"],
        timeout=kwargs.get("timeout", 60),
    )


def _create_paddleocr_vl_qianfan(resolved: ResolvedModel, **kwargs: Any) -> Inference:
    return PaddleOCRVLQianfanInference(
        base_url=kwargs["base_url"],
        api_key=kwargs["api_key"],
        timeout=kwargs.get("timeout", 60),
    )


InferenceFactory.register("ocr", "local", _create_ocr_local)
InferenceFactory.register("ocr", "aistudio", _create_ocr_aistudio)
InferenceFactory.register("ocr", "self_hosted", _create_ocr_self_hosted)

InferenceFactory.register("pp_structurev3", "local", _create_pp_structurev3_local)
InferenceFactory.register("pp_structurev3", "aistudio", _create_pp_structurev3_aistudio)
InferenceFactory.register("pp_structurev3", "qianfan", _create_pp_structurev3_qianfan)
InferenceFactory.register(
    "pp_structurev3", "self_hosted", _create_pp_structurev3_self_hosted
)

InferenceFactory.register("paddleocr_vl", "local", _create_paddleocr_vl_local)
InferenceFactory.register("paddleocr_vl", "aistudio", _create_paddleocr_vl_aistudio)
InferenceFactory.register("paddleocr_vl", "qianfan", _create_paddleocr_vl_qianfan)
InferenceFactory.register(
    "paddleocr_vl", "self_hosted", _create_paddleocr_vl_self_hosted
)


def create_inference(
    resolved: ResolvedModel,
    source: str,
    **kwargs,
) -> Inference:
    return InferenceFactory.create(resolved, source, **kwargs)
