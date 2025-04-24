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


class TextDetection(PaddleXPredictorWrapper):
    def __init__(
        self,
        *,
        limit_side_len=None,
        limit_type=None,
        thresh=None,
        box_thresh=None,
        unclip_ratio=None,
        input_shape=None,
        **kwargs,
    ):
        self._extra_init_args = {
            "limit_side_len": limit_side_len,
            "limit_type": limit_type,
            "thresh": thresh,
            "box_thresh": box_thresh,
            "unclip_ratio": unclip_ratio,
            "input_shape": input_shape,
        }
        super().__init__(**kwargs)

    @property
    def default_model_name(self):
        return "PP-OCRv4_mobile_det"

    @classmethod
    def get_cli_subcommand_executor(cls):
        return TextDetectionSubcommandExecutor()

    def _get_extra_paddlex_predictor_init_args(self):
        return self._extra_init_args


class TextDetectionSubcommandExecutor(PredictorCLISubcommandExecutor):
    @property
    def subparser_name(self):
        return "text_detection"

    def _update_subparser(self, subparser):
        add_simple_inference_args(subparser)

        subparser.add_argument(
            "--limit_side_len",
            type=int,
            help="This sets a limit on the side length of the input image for the model.",
        )
        subparser.add_argument(
            "--limit_type",
            type=str,
            help="This determines how the side length limit is applied to the input image before feeding it into the model.",
        )
        subparser.add_argument(
            "--thresh",
            type=float,
            help="Detection pixel threshold for the model. Pixels with scores greater than this threshold in the output probability map are considered text pixels.",
        )
        subparser.add_argument(
            "--box_thresh",
            type=float,
            help="Detection box threshold for the model. A detection result is considered a text region if the average score of all pixels within the border of the result is greater than this threshold.",
        )
        subparser.add_argument(
            "--unclip_ratio",
            type=float,
            help="Expansion coefficient, which expands the text region using this method. The larger the value, the larger the expansion area.",
        )
        subparser.add_argument(
            "--input_shape",
            nargs=3,
            type=int,
            metavar=("C", "H", "W"),
            help="Input shape of the model.",
        )

    def execute_with_args(self, args):
        params = get_subcommand_args(args)
        perform_simple_inference(TextDetection, params)
