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

from typing import Dict

from ..inference.base import Inference
from .base import Task


class TaskFactory:
    """Factory for creating task instances."""

    _registry: Dict[str, type[Task]] = {}

    @classmethod
    def register(cls, pipeline: str, task_class: type[Task]) -> None:
        """Register a task class.

        Args:
            pipeline: Pipeline name.
            task_class: Task class.
        """
        cls._registry[pipeline] = task_class

    @classmethod
    def create(cls, pipeline: str, inference: Inference) -> Task:
        """Create a task instance.

        Args:
            pipeline: Pipeline name.
            inference: Inference instance.

        Returns:
            Task instance.

        Raises:
            ValueError: If pipeline is not registered.
        """
        if pipeline not in cls._registry:
            raise ValueError(
                f"Unknown pipeline: {pipeline}. Supported: {sorted(cls._registry.keys())}"
            )
        task_class = cls._registry[pipeline]
        return task_class(inference)

    @classmethod
    def list_supported(cls) -> set[str]:
        """List all supported pipelines.

        Returns:
            Set of pipeline names.
        """
        return set(cls._registry.keys())


# Register tasks
from .ocr import OCRTask
from .doc_parsing import PPStructureV3Task, PaddleOCRVLTask

TaskFactory.register("OCR", OCRTask)
TaskFactory.register("PP-StructureV3", PPStructureV3Task)
TaskFactory.register("PaddleOCR-VL", PaddleOCRVLTask)
TaskFactory.register("PaddleOCR-VL-1.5", PaddleOCRVLTask)
TaskFactory.register("PaddleOCR-VL-1.6", PaddleOCRVLTask)


def create_task(pipeline: str, inference: Inference) -> Task:
    """Create a task instance.

    This is a convenience function that delegates to TaskFactory.create.

    Args:
        pipeline: Pipeline name.
        inference: Inference instance.

    Returns:
        Task instance.
    """
    return TaskFactory.create(pipeline, inference)
