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

from .._utils.cli import add_simple_inference_args
from ._doc_vlm import (
    BaseDocVLM,
    BaseDocVLMSubcommandExecutor,
)


class ChartParsing(BaseDocVLM):
    @property
    def default_model_name(self):
        return "PP-Chart2Table"

    @classmethod
    def get_cli_subcommand_executor(cls):
        return ChartParsingSubcommandExecutor()


class ChartParsingSubcommandExecutor(BaseDocVLMSubcommandExecutor):
    @property
    def subparser_name(self):
        return "chart_parsing"

    @property
    def wrapper_cls(self):
        return ChartParsing

    def _update_subparser(self, subparser):
        add_simple_inference_args(
            subparser,
            input_help='Input dict, e.g. `{"image": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/chart_parsing_02.png"}`.',
        )
