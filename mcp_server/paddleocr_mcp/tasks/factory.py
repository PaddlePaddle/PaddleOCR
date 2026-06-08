# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

from ..inference.base import Inference
from ..selection import ResolvedModel
from .base import Task


class TaskFactory:
    _registry: Dict[str, type[Task]] = {}

    @classmethod
    def register(cls, tool: str, task_class: type[Task]) -> None:
        cls._registry[tool] = task_class

    @classmethod
    def create(cls, resolved: ResolvedModel, inference: Inference) -> Task:
        if resolved.tool not in cls._registry:
            raise ValueError(
                f"Unknown tool for model {resolved.model!r}: {resolved.tool}. "
                f"Supported: {sorted(cls._registry.keys())}"
            )
        task_class = cls._registry[resolved.tool]
        return task_class(inference)

    @classmethod
    def list_supported(cls) -> set[str]:
        return set(cls._registry.keys())


from .ocr import OCRTask
from .doc_parsing import PPStructureV3Task, PaddleOCRVLTask

TaskFactory.register("ocr", OCRTask)
TaskFactory.register("pp_structurev3", PPStructureV3Task)
TaskFactory.register("paddleocr_vl", PaddleOCRVLTask)


def create_task(resolved: ResolvedModel, inference: Inference) -> Task:
    return TaskFactory.create(resolved, inference)
