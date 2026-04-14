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

import argparse
from typing import Any, Dict

from .._abstract import CLISubcommandExecutor
from .._utils.cli import (
    add_simple_inference_args,
    get_subcommand_args,
    perform_simple_inference,
)
from .base import PaddleXPredictorWrapper, PredictorCLISubcommandExecutor


class FormulaRecognition(PaddleXPredictorWrapper):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._extra_init_args: Dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    @property
    def default_model_name(self) -> str:
        return "PP-FormulaNet_plus-M"

    @classmethod
    def get_cli_subcommand_executor(cls) -> CLISubcommandExecutor:
        return FormulaRecognitionSubcommandExecutor()

    def _get_extra_paddlex_predictor_init_args(self) -> Dict[str, Any]:
        return self._extra_init_args


class FormulaRecognitionSubcommandExecutor(PredictorCLISubcommandExecutor):
    @property
    def subparser_name(self) -> str:
        return "formula_recognition"

    def _update_subparser(self, subparser: argparse.ArgumentParser) -> None:
        add_simple_inference_args(subparser)

    def execute_with_args(self, args: argparse.Namespace) -> None:
        params = get_subcommand_args(args)
        perform_simple_inference(FormulaRecognition, params)
