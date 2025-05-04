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

from ..utils.cli import (
    add_simple_inference_args,
    get_subcommand_args,
    perform_simple_inference,
)
from .base import PaddleXPredictorWrapper, PredictorCLISubcommandExecutor


class TextRecognition(PaddleXPredictorWrapper):
    def __init__(
        self,
        *,
        input_shape=None,
        **kwargs,
    ):
        self._extra_init_args = {
            "input_shape": input_shape,
        }
        super().__init__(**kwargs)

    @property
    def default_model_name(self):
        return "PP-OCRv4_mobile_rec"

    @classmethod
    def get_cli_subcommand_executor(cls):
        return TextRecognitionSubcommandExecutor()

    def _get_extra_paddlex_predictor_init_args(self):
        return self._extra_init_args


class TextRecognitionSubcommandExecutor(PredictorCLISubcommandExecutor):
    @property
    def subparser_name(self):
        return "text_recognition"

    def _update_subparser(self, subparser):
        add_simple_inference_args(subparser)
        subparser.add_argument(
            "--input_shape",
            nargs=3,
            type=int,
            metavar=("C", "H", "W"),
            help="Input shape of the model.",
        )

    def execute_with_args(self, args):
        params = get_subcommand_args(args)
        perform_simple_inference(TextRecognition, params)
