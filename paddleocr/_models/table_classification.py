# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Type

from .._abstract import CLISubcommandExecutor
from .base import PaddleXPredictorWrapper
from ._image_classification import (
    ImageClassification,
    ImageClassificationSubcommandExecutor,
)


class TableClassification(ImageClassification):
    @property
    def default_model_name(self) -> str:
        return "PP-LCNet_x1_0_table_cls"

    @classmethod
    def get_cli_subcommand_executor(cls) -> CLISubcommandExecutor:
        return TableClassificationSubcommandExecutor()


class TableClassificationSubcommandExecutor(ImageClassificationSubcommandExecutor):
    @property
    def subparser_name(self) -> str:
        return "table_classification"

    @property
    def wrapper_cls(self) -> Type[PaddleXPredictorWrapper]:
        return TableClassification
