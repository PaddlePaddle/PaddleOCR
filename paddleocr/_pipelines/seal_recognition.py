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


class SealRecognition(PaddleXPipelineWrapper):
    def __init__(
        self,
        doc_orientation_classify_model_name=None,
        doc_orientation_classify_model_dir=None,
        doc_unwarping_model_name=None,
        doc_unwarping_model_dir=None,
        layout_detection_model_name=None,
        layout_detection_model_dir=None,
        seal_text_detection_model_name=None,
        seal_text_detection_model_dir=None,
        text_recognition_model_name=None,
        text_recognition_model_dir=None,
        text_recognition_batch_size=None,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_layout_detection=None,
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        seal_det_limit_side_len=None,
        seal_det_limit_type=None,
        seal_det_thresh=None,
        seal_det_box_thresh=None,
        seal_det_unclip_ratio=None,
        seal_rec_score_thresh=None,
        **kwargs,
    ):

        self._params = {
            "doc_orientation_classify_model_name": doc_orientation_classify_model_name,
            "doc_orientation_classify_model_dir": doc_orientation_classify_model_dir,
            "doc_unwarping_model_name": doc_unwarping_model_name,
            "doc_unwarping_model_dir": doc_unwarping_model_dir,
            "layout_detection_model_name": layout_detection_model_name,
            "layout_detection_model_dir": layout_detection_model_dir,
            "seal_text_detection_model_name": seal_text_detection_model_name,
            "seal_text_detection_model_dir": seal_text_detection_model_dir,
            "text_recognition_model_name": text_recognition_model_name,
            "text_recognition_model_dir": text_recognition_model_dir,
            "text_recognition_batch_size": text_recognition_batch_size,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_layout_detection": use_layout_detection,
            "layout_threshold": layout_threshold,
            "layout_nms": layout_nms,
            "layout_unclip_ratio": layout_unclip_ratio,
            "layout_merge_bboxes_mode": layout_merge_bboxes_mode,
            "seal_det_limit_side_len": seal_det_limit_side_len,
            "seal_det_limit_type": seal_det_limit_type,
            "seal_det_thresh": seal_det_thresh,
            "seal_det_box_thresh": seal_det_box_thresh,
            "seal_det_unclip_ratio": seal_det_unclip_ratio,
            "seal_rec_score_thresh": seal_rec_score_thresh,
        }
        super().__init__(**kwargs)

    @property
    def _paddlex_pipeline_name(self):
        return "seal_recognition"

    def predict(
        self,
        input,
        *,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_layout_detection=None,
        layout_det_res=None,
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        seal_det_limit_side_len=None,
        seal_det_limit_type=None,
        seal_det_thresh=None,
        seal_det_box_thresh=None,
        seal_det_unclip_ratio=None,
        seal_rec_score_thresh=None,
        **kwargs,
    ):
        result = []
        for res in self.paddlex_pipeline.predict(
            input,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_layout_detection=use_layout_detection,
            layout_det_res=layout_det_res,
            layout_threshold=layout_threshold,
            layout_nms=layout_nms,
            layout_unclip_ratio=layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,
            seal_det_limit_side_len=seal_det_limit_side_len,
            seal_det_limit_type=seal_det_limit_type,
            seal_det_thresh=seal_det_thresh,
            seal_det_box_thresh=seal_det_box_thresh,
            seal_det_unclip_ratio=seal_det_unclip_ratio,
            seal_rec_score_thresh=seal_rec_score_thresh,
            **kwargs,
        ):
            result.append(res)
        return result

    @classmethod
    def get_cli_subcommand_executor(cls):
        return SealRecognitionCLISubcommandExecutor()

    def _get_paddlex_config_overrides(self):
        STRUCTURE = {
            "SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_name": self._params[
                "doc_orientation_classify_model_name"
            ],
            "SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_dir": self._params[
                "doc_orientation_classify_model_dir"
            ],
            "SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_name": self._params[
                "doc_unwarping_model_name"
            ],
            "SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_dir": self._params[
                "doc_unwarping_model_dir"
            ],
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
            "SubPipelines.DocPreprocessor.use_doc_orientation_classify": self._params[
                "use_doc_orientation_classify"
            ],
            "SubPipelines.DocPreprocessor.use_doc_unwarping": self._params[
                "use_doc_unwarping"
            ],
            "SubPipelines.SealOCR.SubModules.TextDetection.model_name": self._params[
                "seal_text_detection_model_name"
            ],
            "SubPipelines.SealOCR.SubModules.TextDetection.model_dir": self._params[
                "seal_text_detection_model_dir"
            ],
            "SubPipelines.SealOCR.SubModules.TextDetection.limit_side_len": self._params[
                "seal_det_limit_side_len"
            ],
            "SubPipelines.SealOCR.SubModules.TextDetection.limit_type": self._params[
                "seal_det_limit_type"
            ],
            "SubPipelines.SealOCR.SubModules.TextDetection.thresh": self._params[
                "seal_det_thresh"
            ],
            "SubPipelines.SealOCR.SubModules.TextDetection.box_thresh": self._params[
                "seal_det_box_thresh"
            ],
            "SubPipelines.SealOCR.SubModules.TextDetection.unclip_ratio": self._params[
                "seal_det_unclip_ratio"
            ],
            "SubPipelines.SealOCR.SubModules.TextRecognition.model_name": self._params[
                "text_recognition_model_name"
            ],
            "SubPipelines.SealOCR.SubModules.TextRecognition.model_dir": self._params[
                "text_recognition_model_dir"
            ],
            "SubPipelines.SealOCR.SubModules.TextRecognition.batch_size": self._params[
                "text_recognition_batch_size"
            ],
            "SubPipelines.SealOCR.SubModules.TextRecognition.score_thresh": self._params[
                "seal_rec_score_thresh"
            ],
            "use_layout_detection": self._params["use_layout_detection"],
        }
        return create_config_from_structure(STRUCTURE)


class SealRecognitionCLISubcommandExecutor(PipelineCLISubcommandExecutor):
    @property
    def subparser_name(self):
        return "seal_recognition"

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
            "--layout_detection_model_name",
            type=str,
            help="Name of the layout detection model.",
        )
        subparser.add_argument(
            "--layout_detection_model_dir",
            type=str,
            help="Path to the layout detection model directory.",
        )
        subparser.add_argument(
            "--seal_text_detection_model_name",
            type=str,
            help="Name of the seal text detection model.",
        )
        subparser.add_argument(
            "--seal_text_detection_model_dir",
            type=str,
            help="Path to the seal text detection model directory.",
        )
        subparser.add_argument(
            "--text_recognition_model_name",
            type=str,
            help="Name of the text recognition model.",
        )
        subparser.add_argument(
            "--text_recognition_model_dir",
            type=str,
            help="Path to the text recognition model directory.",
        )
        subparser.add_argument(
            "--text_recognition_batch_size",
            type=int,
            help="Batch size for the text recognition model.",
        )
        subparser.add_argument(
            "--use_doc_orientation_classify",
            type=str2bool,
            help="Whether to use the document image orientation classification model.",
        )
        subparser.add_argument(
            "--use_doc_unwarping",
            type=str2bool,
            help="Whether to use the document image unwarping model.",
        )
        subparser.add_argument(
            "--use_layout_detection",
            type=str2bool,
            help="Whether to use the layout detection model.",
        )
        subparser.add_argument(
            "--layout_threshold",
            type=float,
            help="Threshold for layout detection model.",
        )
        subparser.add_argument(
            "--layout_nms",
            type=float,
            help="Non-Maximum Suppression threshold for layout detection.",
        )
        subparser.add_argument(
            "--layout_unclip_ratio",
            type=float,
            help="Layout detection expansion coefficient.",
        )
        subparser.add_argument(
            "--layout_merge_bboxes_mode",
            type=str,
            help="Mode for merging bounding boxes in layout detection.",
        )
        subparser.add_argument(
            "--seal_det_limit_side_len",
            type=int,
            help="This sets a limit on the side length of the input image for the seal text detection model.",
        )
        subparser.add_argument(
            "--seal_det_limit_type",
            type=str,
            help="This determines how the side length limit is applied to the input image before feeding it into the seal text detection model.",
        )
        subparser.add_argument(
            "--seal_det_thresh",
            type=float,
            help="Detection pixel threshold for the seal text detection model. Pixels with scores greater than this threshold in the output probability map are considered text pixels.",
        )
        subparser.add_argument(
            "--seal_det_box_thresh",
            type=float,
            help="Detection box threshold for the seal text detection model. A detection result is considered a text region if the average score of all pixels within the border of the result is greater than this threshold.",
        )
        subparser.add_argument(
            "--seal_det_unclip_ratio",
            type=float,
            help="Seal text detection expansion coefficient, which expands the text region using this method. The larger the value, the larger the expansion area.",
        )
        subparser.add_argument(
            "--seal_rec_score_thresh",
            type=float,
            help="Text recognition threshold. Text results with scores greater than this threshold are retained.",
        )

    def execute_with_args(self, args):
        params = get_subcommand_args(args)

        perform_simple_inference(SealRecognition, params)
