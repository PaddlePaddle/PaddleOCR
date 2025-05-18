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

import abc

from .._utils.cli import (
    add_simple_inference_args,
    get_subcommand_args,
    perform_simple_inference,
    str2bool,
)
from .base import PaddleXPredictorWrapper, PredictorCLISubcommandExecutor


class ObjectDetection(PaddleXPredictorWrapper):
    def __init__(
        self,
        *,
        img_size=None,
        threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        **kwargs,
    ):
        self._extra_init_args = {
            "img_size": img_size,
            "threshold": threshold,
            "layout_nms": layout_nms,
            "layout_unclip_ratio": layout_unclip_ratio,
            "layout_merge_bboxes_mode": layout_merge_bboxes_mode,
        }
        super().__init__(**kwargs)

    def _get_extra_paddlex_predictor_init_args(self):
        return self._extra_init_args


class ObjectDetectionSubcommandExecutor(PredictorCLISubcommandExecutor):
    def _update_subparser(self, subparser):
        add_simple_inference_args(subparser)

        subparser.add_argument(
            "--img_size",
            type=int,
            help="The input image size (w, h).",
        )
        subparser.add_argument(
            "--threshold",
            type=float,
            help="The threshold for filtering out low-confidence predictions.",
        )
        subparser.add_argument(
            "--layout_nms",
            type=str2bool,
            help="Whether to use layout-aware NMS.",
        )
        subparser.add_argument(
            "--layout_unclip_ratio",
            type=float,
            help="The ratio of unclipping the bounding box.",
        )
        subparser.add_argument(
            "--layout_merge_bboxes_mode",
            type=str,
            help="The mode for merging bounding boxes.",
        )

    @property
    @abc.abstractmethod
    def wrapper_cls(self):
        raise NotImplementedError

    def execute_with_args(self, args):
        params = get_subcommand_args(args)
        perform_simple_inference(self.wrapper_cls, params)
