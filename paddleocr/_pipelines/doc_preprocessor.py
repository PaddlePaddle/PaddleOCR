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
    str2bool,
)
from .base import PaddleXPipelineWrapper, PipelineCLISubcommandExecutor
from .utils import create_config_from_structure


class DocPreprocessor(PaddleXPipelineWrapper):
    def __init__(
        self,
        doc_orientation_classify_model_name=None,
        doc_orientation_classify_model_dir=None,
        doc_unwarping_model_name=None,
        doc_unwarping_model_dir=None,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        **kwargs,
    ):

        self._params = {
            "doc_orientation_classify_model_name": doc_orientation_classify_model_name,
            "doc_orientation_classify_model_dir": doc_orientation_classify_model_dir,
            "doc_unwarping_model_name": doc_unwarping_model_name,
            "doc_unwarping_model_dir": doc_unwarping_model_dir,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
        }
        super().__init__(**kwargs)

    @property
    def _paddlex_pipeline_name(self):
        return "doc_preprocessor"

    def predict(
        self,
        input,
        *,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
    ):
        result = []
        for res in self.paddlex_pipeline.predict(
            input,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
        ):
            result.append(res)
        return result

    @classmethod
    def get_cli_subcommand_executor(cls):
        return DocPreprocessorCLISubcommandExecutor()

    def _get_paddlex_config_overrides(self):
        STRUCTURE = {
            "SubModules.DocOrientationClassify.model_name": self._params[
                "doc_orientation_classify_model_name"
            ],
            "SubModules.DocOrientationClassify.model_dir": self._params[
                "doc_orientation_classify_model_dir"
            ],
            "SubModules.DocUnwarping.model_name": self._params[
                "doc_unwarping_model_name"
            ],
            "SubModules.DocUnwarping.model_dir": self._params[
                "doc_unwarping_model_dir"
            ],
            "use_doc_orientation_classify": self._params[
                "use_doc_orientation_classify"
            ],
            "use_doc_unwarping": self._params["use_doc_unwarping"],
        }
        return create_config_from_structure(STRUCTURE)


class DocPreprocessorCLISubcommandExecutor(PipelineCLISubcommandExecutor):
    @property
    def subparser_name(self):
        return "doc_preprocessor"

    def _update_subparser(self, subparser):
        add_simple_inference_args(subparser)

        subparser.add_argument(
            "--doc_orientation_classify_model_name",
            type=str,
            help="Name of the document image orientation classification model.",
        )
        subparser.add_argument(
            "--doc_orientation_classify_model_dir",
            type=str,
            help="Path to the document image orientation classification model directory.",
        )
        subparser.add_argument(
            "--doc_unwarping_model_name",
            type=str,
            help="Name of the document image unwarping model.",
        )
        subparser.add_argument(
            "--doc_unwarping_model_dir",
            type=str,
            help="Path to the document image unwarping model directory.",
        )
        subparser.add_argument(
            "--use_doc_orientation_classify",
            type=str2bool,
            help="Whether to use the document image orientation classification model.",
        )
        subparser.add_argument(
            "--use_doc_unwarping",
            type=str2bool,
            help="Whether to use the text image unwarping model.",
        )

    def execute_with_args(self, args):
        params = get_subcommand_args(args)

        perform_simple_inference(DocPreprocessor, params)
