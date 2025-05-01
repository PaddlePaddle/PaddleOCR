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


class TableRecognitionPipelineV2(PaddleXPipelineWrapper):
    def __init__(
        self,
        layout_detection_model_name=None,
        layout_detection_model_dir=None,
        table_classification_model_name=None,
        table_classification_model_dir=None,
        wired_table_structure_recognition_model_name=None,
        wired_table_structure_recognition_model_dir=None,
        wireless_table_structure_recognition_model_name=None,
        wireless_table_structure_recognition_model_dir=None,
        wired_table_cells_detection_model_name=None,
        wired_table_cells_detection_model_dir=None,
        wireless_table_cells_detection_model_name=None,
        wireless_table_cells_detection_model_dir=None,
        doc_orientation_classify_model_name=None,
        doc_orientation_classify_model_dir=None,
        doc_unwarping_model_name=None,
        doc_unwarping_model_dir=None,
        text_detection_model_name=None,
        text_detection_model_dir=None,
        text_det_limit_side_len=None,
        text_det_limit_type=None,
        text_det_thresh=None,
        text_det_box_thresh=None,
        text_det_unclip_ratio=None,
        text_recognition_model_name=None,
        text_recognition_model_dir=None,
        text_recognition_batch_size=None,
        text_rec_score_thresh=None,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_layout_detection=None,
        use_ocr_model=None,
        **kwargs,
    ):
        params = locals().copy()
        params.pop("self")
        params.pop("kwargs")
        self._params = params

        super().__init__(**kwargs)

    @property
    def _paddlex_pipeline_name(self):
        return "table_recognition_v2"

    def predict(
        self,
        input,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_layout_detection=None,
        use_ocr_model=None,
        overall_ocr_res=None,
        layout_det_res=None,
        text_det_limit_side_len=None,
        text_det_limit_type=None,
        text_det_thresh=None,
        text_det_box_thresh=None,
        text_det_unclip_ratio=None,
        text_rec_score_thresh=None,
        use_table_cells_ocr_results=None,
        use_e2e_wired_table_rec_model=None,
        use_e2e_wireless_table_rec_model=None,
        **kwargs,
    ):
        result = []
        for res in self.paddlex_pipeline.predict(
            input,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_layout_detection=use_layout_detection,
            use_ocr_model=use_ocr_model,
            overall_ocr_res=overall_ocr_res,
            layout_det_res=layout_det_res,
            text_det_limit_side_len=text_det_limit_side_len,
            text_det_limit_type=text_det_limit_type,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_det_unclip_ratio=text_det_unclip_ratio,
            text_rec_score_thresh=text_rec_score_thresh,
            use_table_cells_ocr_results=use_table_cells_ocr_results,
            use_e2e_wired_table_rec_model=use_e2e_wired_table_rec_model,
            use_e2e_wireless_table_rec_model=use_e2e_wireless_table_rec_model,
            **kwargs,
        ):
            result.append(res)
        return result

    @classmethod
    def get_cli_subcommand_executor(cls):
        return TableRecognitionPipelineV2CLISubcommandExecutor()

    def _get_paddlex_config_overrides(self):
        STRUCTURE = {
            "SubPipelines.DocPreprocessor.use_doc_orientation_classify": self._params[
                "use_doc_orientation_classify"
            ],
            "SubPipelines.DocPreprocessor.use_doc_unwarping": self._params[
                "use_doc_unwarping"
            ],
            "use_layout_detection": self._params["use_layout_detection"],
            "use_ocr_model": self._params["use_ocr_model"],
            "SubModules.LayoutDetection.model_name": self._params[
                "layout_detection_model_name"
            ],
            "SubModules.LayoutDetection.model_dir": self._params[
                "layout_detection_model_dir"
            ],
            "SubModules.TableClassification.model_name": self._params[
                "table_classification_model_name"
            ],
            "SubModules.TableClassification.model_dir": self._params[
                "table_classification_model_dir"
            ],
            "SubModules.WiredTableStructureRecognition.model_name": self._params[
                "wired_table_structure_recognition_model_name"
            ],
            "SubModules.WiredTableStructureRecognition.model_dir": self._params[
                "wired_table_structure_recognition_model_dir"
            ],
            "SubModules.WirelessTableStructureRecognition.model_name": self._params[
                "wireless_table_structure_recognition_model_name"
            ],
            "SubModules.WirelessTableStructureRecognition.model_dir": self._params[
                "wireless_table_structure_recognition_model_dir"
            ],
            "SubModules.WiredTableCellsDetection.model_name": self._params[
                "wired_table_cells_detection_model_name"
            ],
            "SubModules.WiredTableCellsDetection.model_dir": self._params[
                "wired_table_cells_detection_model_dir"
            ],
            "SubModules.WirelessTableCellsDetection.model_name": self._params[
                "wireless_table_cells_detection_model_name"
            ],
            "SubModules.WirelessTableCellsDetection.model_dir": self._params[
                "wireless_table_cells_detection_model_dir"
            ],
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
            "SubPipelines.GeneralOCR.SubModules.TextDetection.model_name": self._params[
                "text_detection_model_name"
            ],
            "SubPipelines.GeneralOCR.SubModules.TextDetection.model_dir": self._params[
                "text_detection_model_dir"
            ],
            "SubPipelines.GeneralOCR.SubModules.TextDetection.limit_side_len": self._params[
                "text_det_limit_side_len"
            ],
            "SubPipelines.GeneralOCR.SubModules.TextDetection.limit_type": self._params[
                "text_det_limit_type"
            ],
            "SubPipelines.GeneralOCR.SubModules.TextDetection.thresh": self._params[
                "text_det_thresh"
            ],
            "SubPipelines.GeneralOCR.SubModules.TextDetection.box_thresh": self._params[
                "text_det_box_thresh"
            ],
            "SubPipelines.GeneralOCR.SubModules.TextDetection.unclip_ratio": self._params[
                "text_det_unclip_ratio"
            ],
            "SubPipelines.GeneralOCR.SubModules.TextRecognition.model_name": self._params[
                "text_recognition_model_name"
            ],
            "SubPipelines.GeneralOCR.SubModules.TextRecognition.model_dir": self._params[
                "text_recognition_model_dir"
            ],
            "SubPipelines.GeneralOCR.SubModules.TextRecognition.batch_size": self._params[
                "text_recognition_batch_size"
            ],
            "SubPipelines.GeneralOCR.SubModules.TextRecognition.score_thresh": self._params[
                "text_rec_score_thresh"
            ],
        }
        return create_config_from_structure(STRUCTURE)


class TableRecognitionPipelineV2CLISubcommandExecutor(PipelineCLISubcommandExecutor):
    @property
    def subparser_name(self):
        return "table_recognition_v2"

    def _update_subparser(self, subparser):
        add_simple_inference_args(subparser)

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
            "--table_classification_model_name",
            type=str,
            help="Name of the table classification model.",
        )
        subparser.add_argument(
            "--table_classification_model_dir",
            type=str,
            help="Path to the table classification model directory.",
        )
        subparser.add_argument(
            "--wired_table_structure_recognition_model_name",
            type=str,
            help="Name of the wired table structure recognition model.",
        )
        subparser.add_argument(
            "--wired_table_structure_recognition_model_dir",
            type=str,
            help="Path to the wired table structure recognition model directory.",
        )
        subparser.add_argument(
            "--wireless_table_structure_recognition_model_name",
            type=str,
            help="Name of the wireless table structure recognition model.",
        )
        subparser.add_argument(
            "--wireless_table_structure_recognition_model_dir",
            type=str,
            help="Path to the wired table structure recognition model directory.",
        )
        subparser.add_argument(
            "--wired_table_cells_detection_model_name",
            type=str,
            help="Name of the wired table cells detection model.",
        )
        subparser.add_argument(
            "--wired_table_cells_detection_model_dir",
            type=str,
            help="Path to the wired table cells detection model directory.",
        )
        subparser.add_argument(
            "--wireless_table_cells_detection_model_name",
            type=str,
            help="Name of the wireless table cells detection model.",
        )
        subparser.add_argument(
            "--wireless_table_cells_detection_model_dir",
            type=str,
            help="Path to the wireless table cells detection model directory.",
        )

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
            help="Name of the text image unwarping model.",
        )
        subparser.add_argument(
            "--doc_unwarping_model_dir",
            type=str,
            help="Path to the image unwarping model directory.",
        )

        subparser.add_argument(
            "--text_detection_model_name",
            type=str,
            help="Name of the text detection model.",
        )
        subparser.add_argument(
            "--text_detection_model_dir",
            type=str,
            help="Path to the text detection model directory.",
        )
        subparser.add_argument(
            "--text_det_limit_side_len",
            type=int,
            help="This sets a limit on the side length of the input image for the text detection model.",
        )
        subparser.add_argument(
            "--text_det_limit_type",
            type=str,
            help="This determines how the side length limit is applied to the input image before feeding it into the text deteciton model.",
        )
        subparser.add_argument(
            "--text_det_thresh",
            type=float,
            help="Detection pixel threshold for the text detection model. Pixels with scores greater than this threshold in the output probability map are considered text pixels.",
        )
        subparser.add_argument(
            "--text_det_box_thresh",
            type=float,
            help="Detection box threshold for the text detection model. A detection result is considered a text region if the average score of all pixels within the border of the result is greater than this threshold.",
        )
        subparser.add_argument(
            "--text_det_unclip_ratio",
            type=float,
            help="Text detection expansion coefficient, which expands the text region using this method. The larger the value, the larger the expansion area.",
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
            "--text_rec_score_thresh",
            type=float,
            help="Text recognition threshold used in general OCR. Text results with scores greater than this threshold are retained.",
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
        subparser.add_argument(
            "--use_layout_detection",
            type=str2bool,
            help="Whether to use layout detection.",
        )
        subparser.add_argument(
            "--use_ocr_model",
            type=str2bool,
            help="Whether to use OCR models.",
        )

    def execute_with_args(self, args):
        params = get_subcommand_args(args)
        perform_simple_inference(TableRecognitionPipelineV2, params)
