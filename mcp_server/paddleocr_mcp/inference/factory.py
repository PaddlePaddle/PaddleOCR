# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
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


class InferenceFactory:
    _registry: dict[tuple[str, str], Callable[..., Inference]] = {}

    @classmethod
    def register(
        cls,
        pipeline: str,
        source: str,
        factory_fn: Callable[..., Inference],
    ) -> None:
        cls._registry[(pipeline, source)] = factory_fn

    @classmethod
    def create(
        cls,
        pipeline: str,
        source: str,
        **kwargs,
    ) -> Inference:
        key = (pipeline, source)
        if key not in cls._registry:
            raise ValueError(
                f"Unsupported inference combination: {pipeline} + {source}. "
                f"Supported combinations: {sorted(cls._registry.keys())}"
            )
        factory_fn = cls._registry[key]
        return factory_fn(**kwargs)

    @classmethod
    def list_supported(cls) -> set[tuple[str, str]]:
        return set(cls._registry.keys())


InferenceFactory.register("OCR", "local", OCRLocalInference)
InferenceFactory.register("OCR", "aistudio", OCRAIStudioInference)
InferenceFactory.register("OCR", "self_hosted", OCRSelfHostedInference)

InferenceFactory.register("PP-StructureV3", "local", PPStructureV3LocalInference)
InferenceFactory.register("PP-StructureV3", "aistudio", PPStructureV3AIStudioInference)
InferenceFactory.register("PP-StructureV3", "qianfan", PPStructureV3QianfanInference)
InferenceFactory.register(
    "PP-StructureV3", "self_hosted", PPStructureV3SelfHostedInference
)


def _create_paddleocr_vl_local(version: str) -> Callable[..., Inference]:
    def _factory(**kwargs: Any) -> Inference:
        return PaddleOCRVLLocalInference(**kwargs, version=version)

    return _factory


def _create_paddleocr_vl_aistudio(version: str) -> Callable[..., Inference]:
    def _factory(**kwargs: Any) -> Inference:
        return PaddleOCRVLAIStudioInference(**kwargs, version=version)

    return _factory


for _pipeline, _version in {
    "PaddleOCR-VL": "v1",
    "PaddleOCR-VL-1.5": "v1.5",
    "PaddleOCR-VL-1.6": "v1.6",
}.items():
    InferenceFactory.register(_pipeline, "local", _create_paddleocr_vl_local(_version))
    InferenceFactory.register(
        _pipeline, "aistudio", _create_paddleocr_vl_aistudio(_version)
    )
    InferenceFactory.register(_pipeline, "self_hosted", PaddleOCRVLSelfHostedInference)

InferenceFactory.register("PaddleOCR-VL", "qianfan", PaddleOCRVLQianfanInference)


def create_inference(
    pipeline: str,
    source: str,
    **kwargs,
) -> Inference:
    return InferenceFactory.create(pipeline, source, **kwargs)
