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
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .._abstract import CLISubcommandExecutor
from .._types import InputType, LayoutDetResult, PredictResult
from .._utils.cli import (
    add_simple_inference_args,
    get_subcommand_args,
    perform_simple_inference,
    str2bool,
)
from .base import PaddleXPipelineWrapper, PipelineCLISubcommandExecutor
from .utils import create_config_from_structure


class FormulaRecognitionPipeline(PaddleXPipelineWrapper):
    def __init__(
        self,
        doc_orientation_classify_model_name: Optional[str] = None,
        doc_orientation_classify_model_dir: Optional[str] = None,
        doc_orientation_classify_batch_size: Optional[int] = None,
        doc_unwarping_model_name: Optional[str] = None,
        doc_unwarping_model_dir: Optional[str] = None,
        doc_unwarping_batch_size: Optional[int] = None,
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        layout_detection_model_name: Optional[str] = None,
        layout_detection_model_dir: Optional[str] = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        layout_detection_batch_size: Optional[int] = None,
        use_layout_detection: Optional[bool] = None,
        formula_recognition_model_name: Optional[str] = None,
        formula_recognition_model_dir: Optional[str] = None,
        formula_recognition_batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        params = locals().copy()
        params.pop("self")
        params.pop("kwargs")
        self._params = params

        super().__init__(**kwargs)

    @property
    def _paddlex_pipeline_name(self) -> str:
        return "formula_recognition"

    def predict_iter(
        self,
        input: InputType,
        *,
        use_layout_detection: Optional[bool] = None,
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        layout_det_res: LayoutDetResult = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[PredictResult]:
        return self.paddlex_pipeline.predict(
            input,
            use_layout_detection=use_layout_detection,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            layout_det_res=layout_det_res,
            layout_threshold=layout_threshold,
            layout_nms=layout_nms,
            layout_unclip_ratio=layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,
            **kwargs,
        )

    def predict(
        self,
        input: InputType,
        *,
        use_layout_detection: Optional[bool] = None,
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        layout_det_res: LayoutDetResult = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> List[PredictResult]:
        return list(
            self.predict_iter(
                input,
                use_layout_detection=use_layout_detection,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                layout_det_res=layout_det_res,
                layout_threshold=layout_threshold,
                layout_nms=layout_nms,
                layout_unclip_ratio=layout_unclip_ratio,
                layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                **kwargs,
            )
        )

    @classmethod
    def get_cli_subcommand_executor(cls) -> CLISubcommandExecutor:
        return FormulaRecognitionPipelineCLISubcommandExecutor()

    def _get_paddlex_config_overrides(self) -> Dict[str, Any]:
        STRUCTURE = {
            "use_layout_detection": self._params["use_layout_detection"],
            "SubModules.LayoutDetection.model_name": self._params[
                "layout_detection_model_name"
            ],
            "SubModules.LayoutDetection.model_dir": self._params[
                "layout_detection_model_dir"
            ],
            "SubModules.LayoutDetection.threshold": self._params["layout_threshold"],
            "SubModules.LayoutDetection.layout_nms": self._params["layout_nms"],
            "SubModules.LayoutDetection.layout_unclip_ratio": self._params[
                "layout_unclip_ratio"
            ],
            "SubModules.LayoutDetection.layout_merge_bboxes_mode": self._params[
                "layout_merge_bboxes_mode"
            ],
            "SubModules.LayoutDetection.batch_size": self._params[
                "layout_detection_batch_size"
            ],
            "SubModules.FormulaRecognition.model_name": self._params[
                "formula_recognition_model_name"
            ],
            "SubModules.FormulaRecognition.model_dir": self._params[
                "formula_recognition_model_dir"
            ],
            "SubModules.FormulaRecognition.batch_size": self._params[
                "formula_recognition_batch_size"
            ],
            "SubPipelines.DocPreprocessor.use_doc_orientation_classify": self._params[
                "use_doc_orientation_classify"
            ],
            "SubPipelines.DocPreprocessor.use_doc_unwarping": self._params[
                "use_doc_unwarping"
            ],
            "use_doc_preprocessor": self._params["use_doc_orientation_classify"]
            or self._params["use_doc_unwarping"],
            "SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_name": self._params[
                "doc_orientation_classify_model_name"
            ],
            "SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_dir": self._params[
                "doc_orientation_classify_model_dir"
            ],
            "SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.batch_size": self._params[
                "doc_orientation_classify_batch_size"
            ],
            "SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_name": self._params[
                "doc_unwarping_model_name"
            ],
            "SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_dir": self._params[
                "doc_unwarping_model_dir"
            ],
            "SubPipelines.DocPreprocessor.SubModules.DocUnwarping.batch_size": self._params[
                "doc_unwarping_batch_size"
            ],
        }
        return create_config_from_structure(STRUCTURE)


class FormulaRecognitionPipelineCLISubcommandExecutor(PipelineCLISubcommandExecutor):
    @property
    def subparser_name(self) -> str:
        return "formula_recognition_pipeline"

    def _update_subparser(self, subparser: argparse.ArgumentParser) -> None:
        add_simple_inference_args(subparser)

        subparser.add_argument(
            "--doc_orientation_classify_model_name",
            type=str,
            help="Name of the document image orientation classification model.",
        )
        subparser.add_argument(
            "--doc_orientation_classify_model_dir",
            type=str,
            help="Directory of the document image orientation classification model.",
        )
        subparser.add_argument(
            "--doc_orientation_classify_batch_size",
            type=int,
            help="Batch size for document image orientation classification.",
        )
        subparser.add_argument(
            "--doc_unwarping_model_name",
            type=str,
            help="Name of the document unwarping model.",
        )
        subparser.add_argument(
            "--doc_unwarping_model_dir",
            type=str,
            help="Directory of the document unwarping model.",
        )
        subparser.add_argument(
            "--doc_unwarping_batch_size",
            type=int,
            help="Batch size for document unwarping.",
        )
        subparser.add_argument(
            "--use_doc_orientation_classify",
            type=str2bool,
            help="Use document image orientation classification.",
        )
        subparser.add_argument(
            "--use_doc_unwarping",
            type=str2bool,
            help="Use document unwarping.",
        )
        subparser.add_argument(
            "--layout_detection_model_name",
            type=str,
            help="Name of the layout detection model.",
        )
        subparser.add_argument(
            "--layout_detection_model_dir",
            type=str,
            help="Directory of the layout detection model.",
        )
        subparser.add_argument(
            "--layout_threshold",
            type=float,
            help="Threshold for layout detection.",
        )
        subparser.add_argument(
            "--layout_nms",
            type=str2bool,
            help="Non-maximum suppression for layout detection.",
        )
        subparser.add_argument(
            "--layout_unclip_ratio",
            type=float,
            help="Unclip ratio for layout detection.",
        )
        subparser.add_argument(
            "--layout_merge_bboxes_mode",
            type=str,
            help="Mode for merging bounding boxes in layout detection.",
        )
        subparser.add_argument(
            "--layout_detection_batch_size",
            type=int,
            help="Batch size for layout detection.",
        )
        subparser.add_argument(
            "--use_layout_detection",
            type=str2bool,
            help="Use layout detection.",
        )
        subparser.add_argument(
            "--formula_recognition_model_name",
            type=str,
            help="Name of the formula recognition model.",
        )
        subparser.add_argument(
            "--formula_recognition_model_dir",
            type=str,
            help="Directory of the formula recognition model.",
        )
        subparser.add_argument(
            "--formula_recognition_batch_size",
            type=int,
            help="Batch size for formula recognition.",
        )

    def execute_with_args(self, args: argparse.Namespace) -> None:
        params = get_subcommand_args(args)
        perform_simple_inference(FormulaRecognitionPipeline, params)
