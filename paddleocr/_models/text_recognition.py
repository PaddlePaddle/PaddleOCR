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
from typing import Any, Dict, Optional

from .._abstract import CLISubcommandExecutor
from .._utils.cli import (
    add_simple_inference_args,
    get_subcommand_args,
    perform_simple_inference,
)
from .base import PaddleXPredictorWrapper, PredictorCLISubcommandExecutor


class TextRecognition(PaddleXPredictorWrapper):
    def __init__(
        self,
        *,
        input_shape: Optional[tuple] = None,
        **kwargs: Any,
    ) -> None:
        self._extra_init_args = {
            "input_shape": input_shape,
        }
        super().__init__(**kwargs)

    @property
    def default_model_name(self) -> str:
        return "PP-OCRv5_server_rec"

    @classmethod
    def get_cli_subcommand_executor(cls) -> CLISubcommandExecutor:
        return TextRecognitionSubcommandExecutor()

    def _get_extra_paddlex_predictor_init_args(self) -> Dict[str, Any]:
        return self._extra_init_args


class TextRecognitionSubcommandExecutor(PredictorCLISubcommandExecutor):
    @property
    def subparser_name(self) -> str:
        return "text_recognition"

    def _update_subparser(self, subparser: argparse.ArgumentParser) -> None:
        add_simple_inference_args(subparser)
        subparser.add_argument(
            "--input_shape",
            nargs=3,
            type=int,
            metavar=("C", "H", "W"),
            help="Input shape of the model.",
        )

    def execute_with_args(self, args: argparse.Namespace) -> None:
        params = get_subcommand_args(args)
        perform_simple_inference(TextRecognition, params)
