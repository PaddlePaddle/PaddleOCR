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
from .._types import InputType, PredictResult
from .._utils.cli import (
    get_subcommand_args,
    str2bool,
)
from .._utils.logging import logger
from .base import PaddleXPipelineWrapper, PipelineCLISubcommandExecutor
from .utils import create_config_from_structure


class PPDocTranslation(PaddleXPipelineWrapper):
    def __init__(
        self,
        layout_detection_model_name: Optional[str] = None,
        layout_detection_model_dir: Optional[str] = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        chart_recognition_model_name: Optional[str] = None,
        chart_recognition_model_dir: Optional[str] = None,
        chart_recognition_batch_size: Optional[int] = None,
        region_detection_model_name: Optional[str] = None,
        region_detection_model_dir: Optional[str] = None,
        doc_orientation_classify_model_name: Optional[str] = None,
        doc_orientation_classify_model_dir: Optional[str] = None,
        doc_unwarping_model_name: Optional[str] = None,
        doc_unwarping_model_dir: Optional[str] = None,
        text_detection_model_name: Optional[str] = None,
        text_detection_model_dir: Optional[str] = None,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        textline_orientation_model_name: Optional[str] = None,
        textline_orientation_model_dir: Optional[str] = None,
        textline_orientation_batch_size: Optional[int] = None,
        text_recognition_model_name: Optional[str] = None,
        text_recognition_model_dir: Optional[str] = None,
        text_recognition_batch_size: Optional[int] = None,
        text_rec_score_thresh: Optional[float] = None,
        table_classification_model_name: Optional[str] = None,
        table_classification_model_dir: Optional[str] = None,
        wired_table_structure_recognition_model_name: Optional[str] = None,
        wired_table_structure_recognition_model_dir: Optional[str] = None,
        wireless_table_structure_recognition_model_name: Optional[str] = None,
        wireless_table_structure_recognition_model_dir: Optional[str] = None,
        wired_table_cells_detection_model_name: Optional[str] = None,
        wired_table_cells_detection_model_dir: Optional[str] = None,
        wireless_table_cells_detection_model_name: Optional[str] = None,
        wireless_table_cells_detection_model_dir: Optional[str] = None,
        table_orientation_classify_model_name: Optional[str] = None,
        table_orientation_classify_model_dir: Optional[str] = None,
        seal_text_detection_model_name: Optional[str] = None,
        seal_text_detection_model_dir: Optional[str] = None,
        seal_det_limit_side_len: Optional[int] = None,
        seal_det_limit_type: Optional[str] = None,
        seal_det_thresh: Optional[float] = None,
        seal_det_box_thresh: Optional[float] = None,
        seal_det_unclip_ratio: Optional[float] = None,
        seal_text_recognition_model_name: Optional[str] = None,
        seal_text_recognition_model_dir: Optional[str] = None,
        seal_text_recognition_batch_size: Optional[int] = None,
        seal_rec_score_thresh: Optional[float] = None,
        formula_recognition_model_name: Optional[str] = None,
        formula_recognition_model_dir: Optional[str] = None,
        formula_recognition_batch_size: Optional[int] = None,
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_textline_orientation: Optional[bool] = None,
        use_seal_recognition: Optional[bool] = None,
        use_table_recognition: Optional[bool] = None,
        use_formula_recognition: Optional[bool] = None,
        use_chart_recognition: Optional[bool] = None,
        use_region_detection: Optional[bool] = None,
        chat_bot_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        params = locals().copy()
        params.pop("self")
        params.pop("kwargs")
        self._params = params

        super().__init__(**kwargs)

    @property
    def _paddlex_pipeline_name(self) -> str:
        return "PP-DocTranslation"

    def visual_predict_iter(
        self,
        input: InputType,
        *,
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_textline_orientation: Optional[bool] = None,
        use_seal_recognition: Optional[bool] = None,
        use_table_recognition: Optional[bool] = None,
        use_formula_recognition: Optional[bool] = None,
        use_chart_recognition: Optional[bool] = None,
        use_region_detection: Optional[bool] = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_rec_score_thresh: Optional[float] = None,
        seal_det_limit_side_len: Optional[int] = None,
        seal_det_limit_type: Optional[str] = None,
        seal_det_thresh: Optional[float] = None,
        seal_det_box_thresh: Optional[float] = None,
        seal_det_unclip_ratio: Optional[float] = None,
        seal_rec_score_thresh: Optional[float] = None,
        use_wired_table_cells_trans_to_html: bool = False,
        use_wireless_table_cells_trans_to_html: bool = False,
        use_table_orientation_classify: bool = True,
        use_ocr_results_with_table_cells: bool = True,
        use_e2e_wired_table_rec_model: bool = False,
        use_e2e_wireless_table_rec_model: bool = True,
        **kwargs: Any,
    ) -> Iterator[PredictResult]:
        return self.paddlex_pipeline.visual_predict(
            input,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            use_seal_recognition=use_seal_recognition,
            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
            use_chart_recognition=use_chart_recognition,
            use_region_detection=use_region_detection,
            layout_threshold=layout_threshold,
            layout_nms=layout_nms,
            layout_unclip_ratio=layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,
            text_det_limit_side_len=text_det_limit_side_len,
            text_det_limit_type=text_det_limit_type,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_det_unclip_ratio=text_det_unclip_ratio,
            text_rec_score_thresh=text_rec_score_thresh,
            seal_det_limit_side_len=seal_det_limit_side_len,
            seal_det_limit_type=seal_det_limit_type,
            seal_det_thresh=seal_det_thresh,
            seal_det_box_thresh=seal_det_box_thresh,
            seal_det_unclip_ratio=seal_det_unclip_ratio,
            seal_rec_score_thresh=seal_rec_score_thresh,
            use_wired_table_cells_trans_to_html=use_wired_table_cells_trans_to_html,
            use_wireless_table_cells_trans_to_html=use_wireless_table_cells_trans_to_html,
            use_table_orientation_classify=use_table_orientation_classify,
            use_ocr_results_with_table_cells=use_ocr_results_with_table_cells,
            use_e2e_wired_table_rec_model=use_e2e_wired_table_rec_model,
            use_e2e_wireless_table_rec_model=use_e2e_wireless_table_rec_model,
            **kwargs,
        )

    def visual_predict(
        self,
        input: InputType,
        *,
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_textline_orientation: Optional[bool] = None,
        use_seal_recognition: Optional[bool] = None,
        use_table_recognition: Optional[bool] = None,
        use_formula_recognition: Optional[bool] = None,
        use_chart_recognition: Optional[bool] = None,
        use_region_detection: Optional[bool] = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_rec_score_thresh: Optional[float] = None,
        seal_det_limit_side_len: Optional[int] = None,
        seal_det_limit_type: Optional[str] = None,
        seal_det_thresh: Optional[float] = None,
        seal_det_box_thresh: Optional[float] = None,
        seal_det_unclip_ratio: Optional[float] = None,
        seal_rec_score_thresh: Optional[float] = None,
        use_wired_table_cells_trans_to_html: bool = False,
        use_wireless_table_cells_trans_to_html: bool = False,
        use_table_orientation_classify: bool = True,
        use_ocr_results_with_table_cells: bool = True,
        use_e2e_wired_table_rec_model: bool = False,
        use_e2e_wireless_table_rec_model: bool = True,
        **kwargs: Any,
    ) -> List[PredictResult]:
        return list(
            self.visual_predict_iter(
                input,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                use_textline_orientation=use_textline_orientation,
                use_seal_recognition=use_seal_recognition,
                use_table_recognition=use_table_recognition,
                use_formula_recognition=use_formula_recognition,
                use_chart_recognition=use_chart_recognition,
                use_region_detection=use_region_detection,
                layout_threshold=layout_threshold,
                layout_nms=layout_nms,
                layout_unclip_ratio=layout_unclip_ratio,
                layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                text_det_limit_side_len=text_det_limit_side_len,
                text_det_limit_type=text_det_limit_type,
                text_det_thresh=text_det_thresh,
                text_det_box_thresh=text_det_box_thresh,
                text_det_unclip_ratio=text_det_unclip_ratio,
                text_rec_score_thresh=text_rec_score_thresh,
                seal_det_limit_side_len=seal_det_limit_side_len,
                seal_det_limit_type=seal_det_limit_type,
                seal_det_thresh=seal_det_thresh,
                seal_det_box_thresh=seal_det_box_thresh,
                seal_det_unclip_ratio=seal_det_unclip_ratio,
                seal_rec_score_thresh=seal_rec_score_thresh,
                use_wired_table_cells_trans_to_html=use_wired_table_cells_trans_to_html,
                use_wireless_table_cells_trans_to_html=use_wireless_table_cells_trans_to_html,
                use_table_orientation_classify=use_table_orientation_classify,
                use_ocr_results_with_table_cells=use_ocr_results_with_table_cells,
                use_e2e_wired_table_rec_model=use_e2e_wired_table_rec_model,
                use_e2e_wireless_table_rec_model=use_e2e_wireless_table_rec_model,
                **kwargs,
            )
        )

    def translate_iter(
        self,
        ori_md_info_list: List[Any],
        *,
        target_language: str = "zh",
        chunk_size: int = 5000,
        task_description: Optional[str] = None,
        output_format: Optional[str] = None,
        rules_str: Optional[str] = None,
        few_shot_demo_text_content: Optional[str] = None,
        few_shot_demo_key_value_list: Optional[List[Any]] = None,
        glossary: Optional[str] = None,
        llm_request_interval: float = 0.0,
        chat_bot_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        return self.paddlex_pipeline.translate(
            ori_md_info_list,
            target_language=target_language,
            chunk_size=chunk_size,
            task_description=task_description,
            output_format=output_format,
            rules_str=rules_str,
            few_shot_demo_text_content=few_shot_demo_text_content,
            few_shot_demo_key_value_list=few_shot_demo_key_value_list,
            glossary=glossary,
            llm_request_interval=llm_request_interval,
            chat_bot_config=chat_bot_config,
            **kwargs,
        )

    def translate(
        self,
        ori_md_info_list: List[Any],
        *,
        target_language: str = "zh",
        chunk_size: int = 5000,
        task_description: Optional[str] = None,
        output_format: Optional[str] = None,
        rules_str: Optional[str] = None,
        few_shot_demo_text_content: Optional[str] = None,
        few_shot_demo_key_value_list: Optional[List[Any]] = None,
        glossary: Optional[str] = None,
        llm_request_interval: float = 0.0,
        chat_bot_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Any]:
        return list(
            self.translate_iter(
                ori_md_info_list,
                target_language=target_language,
                chunk_size=chunk_size,
                task_description=task_description,
                output_format=output_format,
                rules_str=rules_str,
                few_shot_demo_text_content=few_shot_demo_text_content,
                few_shot_demo_key_value_list=few_shot_demo_key_value_list,
                glossary=glossary,
                llm_request_interval=llm_request_interval,
                chat_bot_config=chat_bot_config,
                **kwargs,
            )
        )

    def load_from_markdown(self, input: InputType) -> Any:
        return self.paddlex_pipeline.load_from_markdown(input)

    def concatenate_markdown_pages(self, markdown_list: List[str]) -> str:
        return self.paddlex_pipeline.concatenate_markdown_pages(markdown_list)

    @classmethod
    def get_cli_subcommand_executor(cls) -> CLISubcommandExecutor:
        return PPDocTranslationCLISubcommandExecutor()

    def _get_paddlex_config_overrides(self) -> Dict[str, Any]:
        # HACK: We should consider reducing duplication.
        STRUCTURE = {
            "SubPipelines.LayoutParser.SubPipelines.DocPreprocessor.use_doc_orientation_classify": self._params[
                "use_doc_orientation_classify"
            ],
            "SubPipelines.LayoutParser.SubPipelines.DocPreprocessor.use_doc_unwarping": self._params[
                "use_doc_unwarping"
            ],
            "SubPipelines.LayoutParser.use_doc_preprocessor": self._params[
                "use_doc_orientation_classify"
            ]
            or self._params["use_doc_unwarping"],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.use_textline_orientation": self._params[
                "use_textline_orientation"
            ],
            "SubPipelines.LayoutParser.use_seal_recognition": self._params[
                "use_seal_recognition"
            ],
            "SubPipelines.LayoutParser.use_table_recognition": self._params[
                "use_table_recognition"
            ],
            "SubPipelines.LayoutParser.use_formula_recognition": self._params[
                "use_formula_recognition"
            ],
            "SubPipelines.LayoutParser.use_chart_recognition": self._params[
                "use_chart_recognition"
            ],
            "SubPipelines.LayoutParser.use_region_detection": self._params[
                "use_region_detection"
            ],
            "SubPipelines.LayoutParser.SubModules.LayoutDetection.model_name": self._params[
                "layout_detection_model_name"
            ],
            "SubPipelines.LayoutParser.SubModules.LayoutDetection.model_dir": self._params[
                "layout_detection_model_dir"
            ],
            "SubPipelines.LayoutParser.SubModules.LayoutDetection.threshold": self._params[
                "layout_threshold"
            ],
            "SubPipelines.LayoutParser.SubModules.LayoutDetection.layout_nms": self._params[
                "layout_nms"
            ],
            "SubPipelines.LayoutParser.SubModules.LayoutDetection.layout_unclip_ratio": self._params[
                "layout_unclip_ratio"
            ],
            "SubPipelines.LayoutParser.SubModules.LayoutDetection.layout_merge_bboxes_mode": self._params[
                "layout_merge_bboxes_mode"
            ],
            "SubPipelines.LayoutParser.SubModules.ChartRecognition.model_name": self._params[
                "chart_recognition_model_name"
            ],
            "SubPipelines.LayoutParser.SubModules.ChartRecognition.model_dir": self._params[
                "chart_recognition_model_dir"
            ],
            "SubPipelines.LayoutParser.SubModules.ChartRecognition.batch_size": self._params[
                "chart_recognition_batch_size"
            ],
            "SubPipelines.LayoutParser.SubModules.RegionDetection.model_name": self._params[
                "region_detection_model_name"
            ],
            "SubPipelines.LayoutParser.SubModules.RegionDetection.model_dir": self._params[
                "region_detection_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_name": self._params[
                "doc_orientation_classify_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_dir": self._params[
                "doc_orientation_classify_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_name": self._params[
                "doc_unwarping_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_dir": self._params[
                "doc_unwarping_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextDetection.model_name": self._params[
                "text_detection_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextDetection.model_dir": self._params[
                "text_detection_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextDetection.limit_side_len": self._params[
                "text_det_limit_side_len"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextDetection.limit_type": self._params[
                "text_det_limit_type"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextDetection.thresh": self._params[
                "text_det_thresh"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextDetection.box_thresh": self._params[
                "text_det_box_thresh"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextDetection.unclip_ratio": self._params[
                "text_det_unclip_ratio"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextLineOrientation.model_name": self._params[
                "textline_orientation_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextLineOrientation.model_dir": self._params[
                "textline_orientation_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextLineOrientation.batch_size": self._params[
                "textline_orientation_batch_size"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextRecognition.model_name": self._params[
                "text_recognition_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextRecognition.model_dir": self._params[
                "text_recognition_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextRecognition.batch_size": self._params[
                "text_recognition_batch_size"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextRecognition.score_thresh": self._params[
                "text_rec_score_thresh"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubModules.TableClassification.model_name": self._params[
                "table_classification_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubModules.TableClassification.model_dir": self._params[
                "table_classification_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubModules.WiredTableStructureRecognition.model_name": self._params[
                "wired_table_structure_recognition_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubModules.WiredTableStructureRecognition.model_dir": self._params[
                "wired_table_structure_recognition_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubModules.WirelessTableStructureRecognition.model_name": self._params[
                "wireless_table_structure_recognition_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubModules.WirelessTableStructureRecognition.model_dir": self._params[
                "wireless_table_structure_recognition_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubModules.WiredTableCellsDetection.model_name": self._params[
                "wired_table_cells_detection_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubModules.WiredTableCellsDetection.model_dir": self._params[
                "wired_table_cells_detection_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubModules.WirelessTableCellsDetection.model_name": self._params[
                "wireless_table_cells_detection_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubModules.WirelessTableCellsDetection.model_dir": self._params[
                "wireless_table_cells_detection_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubModules.TableOrientationClassify.model_name": self._params[
                "table_orientation_classify_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubModules.TableOrientationClassify.model_dir": self._params[
                "table_orientation_classify_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextDetection.model_name": self._params[
                "text_detection_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextDetection.model_dir": self._params[
                "text_detection_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextDetection.limit_side_len": self._params[
                "text_det_limit_side_len"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextDetection.limit_type": self._params[
                "text_det_limit_type"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextDetection.thresh": self._params[
                "text_det_thresh"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextDetection.box_thresh": self._params[
                "text_det_box_thresh"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextDetection.unclip_ratio": self._params[
                "text_det_unclip_ratio"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextLineOrientation.model_name": self._params[
                "textline_orientation_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextLineOrientation.model_dir": self._params[
                "textline_orientation_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextLineOrientation.batch_size": self._params[
                "textline_orientation_batch_size"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextRecognition.model_name": self._params[
                "text_recognition_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextRecognition.model_dir": self._params[
                "text_recognition_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextRecognition.batch_size": self._params[
                "text_recognition_batch_size"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextRecognition.score_thresh": self._params[
                "text_rec_score_thresh"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.model_name": self._params[
                "seal_text_detection_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.model_dir": self._params[
                "seal_text_detection_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.limit_side_len": self._params[
                "text_det_limit_side_len"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.limit_type": self._params[
                "seal_det_limit_type"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.thresh": self._params[
                "seal_det_thresh"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.box_thresh": self._params[
                "seal_det_box_thresh"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.unclip_ratio": self._params[
                "seal_det_unclip_ratio"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextRecognition.model_name": self._params[
                "seal_text_recognition_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextRecognition.model_dir": self._params[
                "seal_text_recognition_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextRecognition.batch_size": self._params[
                "seal_text_recognition_batch_size"
            ],
            "SubPipelines.LayoutParser.SubPipelines.FormulaRecognition.SubModules.FormulaRecognition.model_name": self._params[
                "formula_recognition_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.FormulaRecognition.SubModules.FormulaRecognition.model_dir": self._params[
                "formula_recognition_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.FormulaRecognition.SubModules.FormulaRecognition.batch_size": self._params[
                "formula_recognition_batch_size"
            ],
            "SubModules.LLM_Chat": self._params["chat_bot_config"],
        }
        return create_config_from_structure(STRUCTURE)


class PPDocTranslationCLISubcommandExecutor(PipelineCLISubcommandExecutor):
    @property
    def subparser_name(self) -> str:
        return "pp_doctranslation"

    def _update_subparser(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "-i",
            "--input",
            type=str,
            required=True,
            help="Input path or URL.",
        )
        subparser.add_argument(
            "--save_path",
            type=str,
            help="Path to the output directory.",
        )

        subparser.add_argument(
            "--target_language",
            type=str,
            default="zh",
            help="Target language.",
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
            "--layout_threshold",
            type=float,
            help="Score threshold for the layout detection model.",
        )
        subparser.add_argument(
            "--layout_nms",
            type=str2bool,
            help="Whether to use NMS in layout detection.",
        )
        subparser.add_argument(
            "--layout_unclip_ratio",
            type=float,
            help="Expansion coefficient for layout detection.",
        )
        subparser.add_argument(
            "--layout_merge_bboxes_mode",
            type=str,
            help="Overlapping box filtering method.",
        )

        subparser.add_argument(
            "--chart_recognition_model_name",
            type=str,
            help="Name of the chart recognition model.",
        )
        subparser.add_argument(
            "--chart_recognition_model_dir",
            type=str,
            help="Path to the chart recognition model directory.",
        )
        subparser.add_argument(
            "--chart_recognition_batch_size",
            type=int,
            help="Batch size for the chart recognition model.",
        )

        subparser.add_argument(
            "--region_detection_model_name",
            type=str,
            help="Name of the region detection model.",
        )
        subparser.add_argument(
            "--region_detection_model_dir",
            type=str,
            help="Path to the region detection model directory.",
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
            "--textline_orientation_model_name",
            type=str,
            help="Name of the text line orientation classification model.",
        )
        subparser.add_argument(
            "--textline_orientation_model_dir",
            type=str,
            help="Path to the text line orientation classification directory.",
        )
        subparser.add_argument(
            "--textline_orientation_batch_size",
            type=int,
            help="Batch size for the text line orientation classification model.",
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
            "--seal_det_limit_side_len",
            type=int,
            help="This sets a limit on the side length of the input image for the seal text detection model.",
        )
        subparser.add_argument(
            "--seal_det_limit_type",
            type=str,
            help="This determines how the side length limit is applied to the input image before feeding it into the seal text deteciton model.",
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
            "--seal_text_recognition_model_name",
            type=str,
            help="Name of the seal text recognition model.",
        )
        subparser.add_argument(
            "--seal_text_recognition_model_dir",
            type=str,
            help="Path to the seal text recognition model directory.",
        )
        subparser.add_argument(
            "--seal_text_recognition_batch_size",
            type=int,
            help="Batch size for the seal text recognition model.",
        )
        subparser.add_argument(
            "--seal_rec_score_thresh",
            type=float,
            help="Seal text recognition threshold. Text results with scores greater than this threshold are retained.",
        )

        subparser.add_argument(
            "--formula_recognition_model_name",
            type=str,
            help="Name of the formula recognition model.",
        )
        subparser.add_argument(
            "--formula_recognition_model_dir",
            type=str,
            help="Path to the formula recognition model directory.",
        )
        subparser.add_argument(
            "--formula_recognition_batch_size",
            type=int,
            help="Batch size for the formula recognition model.",
        )

        subparser.add_argument(
            "--use_doc_orientation_classify",
            type=str2bool,
            help="Whether to use document image orientation classification.",
        )
        subparser.add_argument(
            "--use_doc_unwarping",
            type=str2bool,
            help="Whether to use text image unwarping.",
        )
        subparser.add_argument(
            "--use_textline_orientation",
            type=str2bool,
            help="Whether to use text line orientation classification.",
        )
        subparser.add_argument(
            "--use_seal_recognition",
            type=str2bool,
            help="Whether to use seal recognition.",
        )
        subparser.add_argument(
            "--use_table_recognition",
            type=str2bool,
            help="Whether to use table recognition.",
        )
        subparser.add_argument(
            "--use_formula_recognition",
            type=str2bool,
            help="Whether to use formula recognition.",
        )
        subparser.add_argument(
            "--use_chart_recognition",
            type=str2bool,
            help="Whether to use chart recognition.",
        )
        subparser.add_argument(
            "--use_region_detection",
            type=str2bool,
            help="Whether to use region detection.",
        )

        # FIXME: Passing API key through CLI is not secure; consider using
        # environment variables.
        subparser.add_argument(
            "--qianfan_api_key",
            type=str,
            help="Configuration for the embedding model.",
        )

    def execute_with_args(self, args: argparse.Namespace) -> None:
        params = get_subcommand_args(args)
        input = params.pop("input")
        target_language = params.pop("target_language")
        save_path = params.pop("save_path")
        qianfan_api_key = params.pop("qianfan_api_key")
        if qianfan_api_key is not None:
            params["chat_bot_config"] = {
                "module_name": "chat_bot",
                "model_name": "ernie-3.5-8k",
                "base_url": "https://qianfan.baidubce.com/v2",
                "api_type": "openai",
                "api_key": qianfan_api_key,
            }

        chatocr = PPDocTranslation(**params)

        logger.info("Start analyzing images")
        result_visual = chatocr.visual_predict_iter(input)

        ori_md_info_list = []
        for res in result_visual:
            ori_md_info_list.append(res["layout_parsing_result"].markdown)
            if save_path:
                res["layout_parsing_result"].save_all(save_path)

        logger.info("Start translation")
        result_translate = chatocr.translate_iter(
            ori_md_info_list,
            target_language=target_language,
        )

        for trans_res in result_translate:
            trans_res.print()
            if save_path:
                trans_res.save_to_markdown(save_path)
