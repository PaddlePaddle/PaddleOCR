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

from paddlex.utils.pipeline_arguments import custom_type

from .._utils.cli import (
    add_simple_inference_args,
    get_subcommand_args,
    perform_simple_inference,
)
from .base import PaddleXPipelineWrapper, PipelineCLISubcommandExecutor
from .utils import create_config_from_structure


class DocUnderstanding(PaddleXPipelineWrapper):
    def __init__(
        self,
        doc_understanding_model_name=None,
        doc_understanding_model_dir=None,
        doc_understanding_batch_size=None,
        **kwargs,
    ):

        self._params = {
            "doc_understanding_model_name": doc_understanding_model_name,
            "doc_understanding_model_dir": doc_understanding_model_dir,
            "doc_understanding_batch_size": doc_understanding_batch_size,
        }
        super().__init__(**kwargs)

    @property
    def _paddlex_pipeline_name(self):
        return "doc_understanding"

    def predict_iter(self, input, **kwargs):
        return self.paddlex_pipeline.predict(input, **kwargs)

    def predict(
        self,
        input,
        **kwargs,
    ):
        return list(self.predict_iter(input, **kwargs))

    @classmethod
    def get_cli_subcommand_executor(cls):
        return DocUnderstandingCLISubcommandExecutor()

    def _get_paddlex_config_overrides(self):
        STRUCTURE = {
            "SubModules.DocUnderstanding.model_name": self._params[
                "doc_understanding_model_name"
            ],
            "SubModules.DocUnderstanding.model_dir": self._params[
                "doc_understanding_model_dir"
            ],
            "SubModules.DocUnderstanding.batch_size": self._params[
                "doc_understanding_batch_size"
            ],
        }
        return create_config_from_structure(STRUCTURE)


class DocUnderstandingCLISubcommandExecutor(PipelineCLISubcommandExecutor):
    input_validator = staticmethod(custom_type(dict))

    @property
    def subparser_name(self):
        return "doc_understanding"

    def _update_subparser(self, subparser):
        add_simple_inference_args(
            subparser,
            input_help='Input dict, e.g. `{"image": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/medal_table.png", "query": "Recognize this table"}`.',
        )

        subparser.add_argument(
            "--doc_understanding_model_name",
            type=str,
            help="Name of the document understanding model.",
        )
        subparser.add_argument(
            "--doc_understanding_model_dir",
            type=str,
            help="Path to the document understanding model directory.",
        )
        subparser.add_argument(
            "--doc_understanding_batch_size",
            type=str,
            help="Batch size for the document understanding model.",
        )

    def execute_with_args(self, args):
        params = get_subcommand_args(args)
        params["input"] = self.input_validator(params["input"])
        perform_simple_inference(DocUnderstanding, params)
