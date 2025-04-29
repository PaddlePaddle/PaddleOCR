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
from ..utils.logging import logger
from .base import PaddleXPipelineWrapper, PipelineCLISubcommandExecutor
from .utils import create_config_from_structure


class FormulaRecognitionPipeline(PaddleXPipelineWrapper):
    def __init__(
        self,
        doc_orientation_classify_model_name=None,
        doc_orientation_classify_model_dir=None,
        doc_orientation_classify_batch_size=None,
        doc_unwarping_model_name=None,
        doc_unwarping_model_dir=None,
        doc_unwarping_batch_size=None,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        layout_detection_model_name=None,
        layout_detection_model_dir=None,
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        layout_detection_batch_size=None,
        use_layout_detection=None,
        formula_recognition_model_name=None,
        formula_recognition_model_dir=None,
        formula_recognition_batch_size=None,
        **kwargs,
    ):
        params = locals().copy()
        params.pop("self")
        params.pop("kwargs")
        self._params = params

        super().__init__(**kwargs)

    @property
    def _paddlex_pipeline_name(self):
        return "formula_recognition"

    def predict(
        self,
        input,
        *,
        use_layout_detection=None,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        layout_det_res=None,
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        **kwargs,
    ):
        result = []
        for res in self.paddlex_pipeline.predict(
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
        ):
            result.append(res)
        return result

    @classmethod
    def get_cli_subcommand_executor(cls):
        return FormulaRecognitionPipelineCLISubcommandExecutor()

    def _get_paddlex_config_overrides(self):
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
    def subparser_name(self):
        return "formula_recognition"

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

    def execute_with_args(self, args):
        params = get_subcommand_args(args)
        perform_simple_inference(FormulaRecognitionPipeline, params)
