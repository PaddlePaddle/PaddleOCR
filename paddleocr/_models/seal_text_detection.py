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
from typing import Any

from .._abstract import CLISubcommandExecutor
from .._utils.cli import (
    add_simple_inference_args,
    get_subcommand_args,
    perform_simple_inference,
)
from .base import PaddleXPredictorWrapper, PredictorCLISubcommandExecutor
from ._text_detection import TextDetectionMixin, TextDetectionSubcommandExecutorMixin


class SealTextDetection(TextDetectionMixin, PaddleXPredictorWrapper):
    @property
    def default_model_name(self) -> str:
        return "PP-OCRv4_mobile_seal_det"

    @classmethod
    def get_cli_subcommand_executor(cls) -> CLISubcommandExecutor:
        return SealTextDetectionSubcommandExecutor()


class SealTextDetectionSubcommandExecutor(
    TextDetectionSubcommandExecutorMixin, PredictorCLISubcommandExecutor
):
    @property
    def subparser_name(self) -> str:
        return "seal_text_detection"

    def _update_subparser(self, subparser: argparse.ArgumentParser) -> None:
        add_simple_inference_args(subparser)
        self._add_text_detection_args(subparser)

    def execute_with_args(self, args: argparse.Namespace) -> None:
        params = get_subcommand_args(args)
        perform_simple_inference(SealTextDetection, params)
