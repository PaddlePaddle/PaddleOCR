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

import warnings
from .._utils.cli import (
    add_simple_inference_args,
    get_subcommand_args,
    perform_simple_inference,
    str2bool,
)
from .base import PaddleXPipelineWrapper, PipelineCLISubcommandExecutor
from .utils import create_config_from_structure

_SUPPORTED_OCR_VERSIONS = ["PP-OCRv3", "PP-OCRv4", "PP-OCRv5"]


class PPStructureV3(PaddleXPipelineWrapper):
    def __init__(
        self,
        layout_detection_model_name=None,
        layout_detection_model_dir=None,
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        chart_recognition_model_name=None,
        chart_recognition_model_dir=None,
        chart_recognition_batch_size=None,
        region_detection_model_name=None,
        region_detection_model_dir=None,
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
        textline_orientation_model_name=None,
        textline_orientation_model_dir=None,
        textline_orientation_batch_size=None,
        text_recognition_model_name=None,
        text_recognition_model_dir=None,
        text_recognition_batch_size=None,
        text_rec_score_thresh=None,
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
        table_orientation_classify_model_name=None,
        table_orientation_classify_model_dir=None,
        seal_text_detection_model_name=None,
        seal_text_detection_model_dir=None,
        seal_det_limit_side_len=None,
        seal_det_limit_type=None,
        seal_det_thresh=None,
        seal_det_box_thresh=None,
        seal_det_unclip_ratio=None,
        seal_text_recognition_model_name=None,
        seal_text_recognition_model_dir=None,
        seal_text_recognition_batch_size=None,
        seal_rec_score_thresh=None,
        formula_recognition_model_name=None,
        formula_recognition_model_dir=None,
        formula_recognition_batch_size=None,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_textline_orientation=None,
        use_seal_recognition=None,
        use_table_recognition=None,
        use_formula_recognition=None,
        use_chart_recognition=None,
        use_region_detection=None,
        lang=None,
        ocr_version=None,
        **kwargs,
    ):
        if ocr_version is not None and ocr_version not in _SUPPORTED_OCR_VERSIONS:
            raise ValueError(
                f"Invalid OCR version: {ocr_version}. Supported values are {_SUPPORTED_OCR_VERSIONS}."
            )

        if all(
            map(
                lambda p: p is None,
                (
                    text_detection_model_name,
                    text_detection_model_dir,
                    text_recognition_model_name,
                    text_recognition_model_dir,
                ),
            )
        ):
            if lang is not None or ocr_version is not None:
                det_model_name, rec_model_name = self._get_ocr_model_names(
                    lang, ocr_version
                )
                if det_model_name is None or rec_model_name is None:
                    raise ValueError(
                        f"No models are available for the language {repr(lang)} and OCR version {repr(ocr_version)}."
                    )
                text_detection_model_name = det_model_name
                text_recognition_model_name = rec_model_name
        else:
            if lang is not None or ocr_version is not None:
                warnings.warn(
                    "`lang` and `ocr_version` will be ignored when model names or model directories are not `None`.",
                    stacklevel=2,
                )
        params = locals().copy()
        params["text_detection_model_name"] = text_detection_model_name
        params["text_recognition_model_name"] = text_recognition_model_name
        params.pop("self")
        params.pop("kwargs")
        self._params = params

        super().__init__(**kwargs)

    @property
    def _paddlex_pipeline_name(self):
        return "PP-StructureV3"

    def predict_iter(
        self,
        input,
        *,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_textline_orientation=None,
        use_seal_recognition=None,
        use_table_recognition=None,
        use_formula_recognition=None,
        use_chart_recognition=None,
        use_region_detection=None,
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        text_det_limit_side_len=None,
        text_det_limit_type=None,
        text_det_thresh=None,
        text_det_box_thresh=None,
        text_det_unclip_ratio=None,
        text_rec_score_thresh=None,
        seal_det_limit_side_len=None,
        seal_det_limit_type=None,
        seal_det_thresh=None,
        seal_det_box_thresh=None,
        seal_det_unclip_ratio=None,
        seal_rec_score_thresh=None,
        use_wired_table_cells_trans_to_html=False,
        use_wireless_table_cells_trans_to_html=False,
        use_table_orientation_classify=True,
        use_ocr_results_with_table_cells=True,
        use_e2e_wired_table_rec_model=False,
        use_e2e_wireless_table_rec_model=True,
        **kwargs,
    ):
        return self.paddlex_pipeline.predict(
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

    def predict(
        self,
        input,
        *,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_textline_orientation=None,
        use_seal_recognition=None,
        use_table_recognition=None,
        use_formula_recognition=None,
        use_chart_recognition=None,
        use_region_detection=None,
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        text_det_limit_side_len=None,
        text_det_limit_type=None,
        text_det_thresh=None,
        text_det_box_thresh=None,
        text_det_unclip_ratio=None,
        text_rec_score_thresh=None,
        seal_det_limit_side_len=None,
        seal_det_limit_type=None,
        seal_det_thresh=None,
        seal_det_box_thresh=None,
        seal_det_unclip_ratio=None,
        seal_rec_score_thresh=None,
        use_wired_table_cells_trans_to_html=False,
        use_wireless_table_cells_trans_to_html=False,
        use_table_orientation_classify=True,
        use_ocr_results_with_table_cells=True,
        use_e2e_wired_table_rec_model=False,
        use_e2e_wireless_table_rec_model=True,
        **kwargs,
    ):
        return list(
            self.predict_iter(
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

    def concatenate_markdown_pages(self, markdown_list):
        return self.paddlex_pipeline.concatenate_markdown_pages(markdown_list)

    @classmethod
    def get_cli_subcommand_executor(cls):
        return PPStructureV3CLISubcommandExecutor()

    def _get_paddlex_config_overrides(self):
        STRUCTURE = {
            "SubPipelines.DocPreprocessor.use_doc_orientation_classify": self._params[
                "use_doc_orientation_classify"
            ],
            "SubPipelines.DocPreprocessor.use_doc_unwarping": self._params[
                "use_doc_unwarping"
            ],
            "use_doc_preprocessor": self._params["use_doc_orientation_classify"]
            or self._params["use_doc_unwarping"],
            "SubPipelines.GeneralOCR.use_textline_orientation": self._params[
                "use_textline_orientation"
            ],
            "use_seal_recognition": self._params["use_seal_recognition"],
            "use_table_recognition": self._params["use_table_recognition"],
            "use_formula_recognition": self._params["use_formula_recognition"],
            "use_chart_recognition": self._params["use_chart_recognition"],
            "use_region_detection": self._params["use_region_detection"],
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
            "SubModules.ChartRecognition.model_name": self._params[
                "chart_recognition_model_name"
            ],
            "SubModules.ChartRecognition.model_dir": self._params[
                "chart_recognition_model_dir"
            ],
            "SubModules.ChartRecognition.batch_size": self._params[
                "chart_recognition_batch_size"
            ],
            "SubModules.RegionDetection.model_name": self._params[
                "region_detection_model_name"
            ],
            "SubModules.RegionDetection.model_dir": self._params[
                "region_detection_model_dir"
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
            "SubPipelines.GeneralOCR.SubModules.TextLineOrientation.model_name": self._params[
                "textline_orientation_model_name"
            ],
            "SubPipelines.GeneralOCR.SubModules.TextLineOrientation.model_dir": self._params[
                "textline_orientation_model_dir"
            ],
            "SubPipelines.GeneralOCR.SubModules.TextLineOrientation.batch_size": self._params[
                "textline_orientation_batch_size"
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
            "SubPipelines.TableRecognition.SubModules.TableClassification.model_name": self._params[
                "table_classification_model_name"
            ],
            "SubPipelines.TableRecognition.SubModules.TableClassification.model_dir": self._params[
                "table_classification_model_dir"
            ],
            "SubPipelines.TableRecognition.SubModules.WiredTableStructureRecognition.model_name": self._params[
                "wired_table_structure_recognition_model_name"
            ],
            "SubPipelines.TableRecognition.SubModules.WiredTableStructureRecognition.model_dir": self._params[
                "wired_table_structure_recognition_model_dir"
            ],
            "SubPipelines.TableRecognition.SubModules.WirelessTableStructureRecognition.model_name": self._params[
                "wireless_table_structure_recognition_model_name"
            ],
            "SubPipelines.TableRecognition.SubModules.WirelessTableStructureRecognition.model_dir": self._params[
                "wireless_table_structure_recognition_model_dir"
            ],
            "SubPipelines.TableRecognition.SubModules.WiredTableCellsDetection.model_name": self._params[
                "wired_table_cells_detection_model_name"
            ],
            "SubPipelines.TableRecognition.SubModules.WiredTableCellsDetection.model_dir": self._params[
                "wired_table_cells_detection_model_dir"
            ],
            "SubPipelines.TableRecognition.SubModules.WirelessTableCellsDetection.model_name": self._params[
                "wireless_table_cells_detection_model_name"
            ],
            "SubPipelines.TableRecognition.SubModules.WirelessTableCellsDetection.model_dir": self._params[
                "wireless_table_cells_detection_model_dir"
            ],
            "SubPipelines.TableRecognition.SubModules.TableOrientationClassify.model_name": self._params[
                "table_orientation_classify_model_name"
            ],
            "SubPipelines.TableRecognition.SubModules.TableOrientationClassify.model_dir": self._params[
                "table_orientation_classify_model_dir"
            ],
            "SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextDetection.model_name": self._params[
                "text_detection_model_name"
            ],
            "SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextDetection.model_dir": self._params[
                "text_detection_model_dir"
            ],
            "SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextDetection.limit_side_len": self._params[
                "text_det_limit_side_len"
            ],
            "SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextDetection.limit_type": self._params[
                "text_det_limit_type"
            ],
            "SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextDetection.thresh": self._params[
                "text_det_thresh"
            ],
            "SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextDetection.box_thresh": self._params[
                "text_det_box_thresh"
            ],
            "SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextDetection.unclip_ratio": self._params[
                "text_det_unclip_ratio"
            ],
            "SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextLineOrientation.model_name": self._params[
                "textline_orientation_model_name"
            ],
            "SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextLineOrientation.model_dir": self._params[
                "textline_orientation_model_dir"
            ],
            "SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextLineOrientation.batch_size": self._params[
                "textline_orientation_batch_size"
            ],
            "SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextRecognition.model_name": self._params[
                "text_recognition_model_name"
            ],
            "SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextRecognition.model_dir": self._params[
                "text_recognition_model_dir"
            ],
            "SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextRecognition.batch_size": self._params[
                "text_recognition_batch_size"
            ],
            "SubPipelines.TableRecognition.SubPipelines.GeneralOCR.SubModules.TextRecognition.score_thresh": self._params[
                "text_rec_score_thresh"
            ],
            "SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.model_name": self._params[
                "seal_text_detection_model_name"
            ],
            "SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.model_dir": self._params[
                "seal_text_detection_model_dir"
            ],
            "SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.limit_side_len": self._params[
                "text_det_limit_side_len"
            ],
            "SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.limit_type": self._params[
                "seal_det_limit_type"
            ],
            "SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.thresh": self._params[
                "seal_det_thresh"
            ],
            "SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.box_thresh": self._params[
                "seal_det_box_thresh"
            ],
            "SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.unclip_ratio": self._params[
                "seal_det_unclip_ratio"
            ],
            "SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextRecognition.model_name": self._params[
                "seal_text_recognition_model_name"
            ],
            "SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextRecognition.model_dir": self._params[
                "seal_text_recognition_model_dir"
            ],
            "SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextRecognition.batch_size": self._params[
                "seal_text_recognition_batch_size"
            ],
            "SubPipelines.FormulaRecognition.SubModules.FormulaRecognition.model_name": self._params[
                "formula_recognition_model_name"
            ],
            "SubPipelines.FormulaRecognition.SubModules.FormulaRecognition.model_dir": self._params[
                "formula_recognition_model_dir"
            ],
            "SubPipelines.FormulaRecognition.SubModules.FormulaRecognition.batch_size": self._params[
                "formula_recognition_batch_size"
            ],
        }
        return create_config_from_structure(STRUCTURE)

    def _get_ocr_model_names(self, lang, ppocr_version):
        LATIN_LANGS = [
            "af",
            "az",
            "bs",
            "cs",
            "cy",
            "da",
            "de",
            "es",
            "et",
            "fr",
            "ga",
            "hr",
            "hu",
            "id",
            "is",
            "it",
            "ku",
            "la",
            "lt",
            "lv",
            "mi",
            "ms",
            "mt",
            "nl",
            "no",
            "oc",
            "pi",
            "pl",
            "pt",
            "ro",
            "rs_latin",
            "sk",
            "sl",
            "sq",
            "sv",
            "sw",
            "tl",
            "tr",
            "uz",
            "vi",
            "french",
            "german",
        ]
        ARABIC_LANGS = ["ar", "fa", "ug", "ur"]
        ESLAV_LANGS = ["ru", "be", "uk"]
        CYRILLIC_LANGS = [
            "ru",
            "rs_cyrillic",
            "be",
            "bg",
            "uk",
            "mn",
            "abq",
            "ady",
            "kbd",
            "ava",
            "dar",
            "inh",
            "che",
            "lbe",
            "lez",
            "tab",
        ]
        DEVANAGARI_LANGS = [
            "hi",
            "mr",
            "ne",
            "bh",
            "mai",
            "ang",
            "bho",
            "mah",
            "sck",
            "new",
            "gom",
            "sa",
            "bgc",
        ]
        SPECIFIC_LANGS = [
            "ch",
            "en",
            "korean",
            "japan",
            "chinese_cht",
            "te",
            "ka",
            "ta",
        ]

        if lang is None:
            lang = "ch"

        if ppocr_version is None:
            if (
                lang
                in ["ch", "chinese_cht", "en", "japan", "korean", "th", "el"]
                + LATIN_LANGS
                + ESLAV_LANGS
            ):
                ppocr_version = "PP-OCRv5"
            elif lang in (
                LATIN_LANGS
                + ARABIC_LANGS
                + CYRILLIC_LANGS
                + DEVANAGARI_LANGS
                + SPECIFIC_LANGS
            ):
                ppocr_version = "PP-OCRv3"
            else:
                # Unknown language specified
                return None, None

        if ppocr_version == "PP-OCRv5":
            rec_lang, rec_model_name = None, None
            if lang in ("ch", "chinese_cht", "en", "japan"):
                rec_model_name = "PP-OCRv5_server_rec"
            elif lang in LATIN_LANGS:
                rec_lang = "latin"
            elif lang in ESLAV_LANGS:
                rec_lang = "eslav"
            elif lang == "korean":
                rec_lang = "korean"
            elif lang == "th":
                rec_lang = "th"
            elif lang == "el":
                rec_lang = "el"

            if rec_lang is not None:
                rec_model_name = f"{rec_lang}_PP-OCRv5_mobile_rec"
            return "PP-OCRv5_server_det", rec_model_name

        elif ppocr_version == "PP-OCRv4":
            if lang == "ch":
                return "PP-OCRv4_mobile_det", "PP-OCRv4_mobile_rec"
            elif lang == "en":
                return "PP-OCRv4_mobile_det", "en_PP-OCRv4_mobile_rec"
            else:
                return None, None
        else:
            # PP-OCRv3
            rec_lang = None
            if lang in LATIN_LANGS:
                rec_lang = "latin"
            elif lang in ARABIC_LANGS:
                rec_lang = "arabic"
            elif lang in CYRILLIC_LANGS:
                rec_lang = "cyrillic"
            elif lang in DEVANAGARI_LANGS:
                rec_lang = "devanagari"
            else:
                if lang in SPECIFIC_LANGS:
                    rec_lang = lang

            rec_model_name = None
            if rec_lang == "ch":
                rec_model_name = "PP-OCRv3_mobile_rec"
            elif rec_lang is not None:
                rec_model_name = f"{rec_lang}_PP-OCRv3_mobile_rec"
            return "PP-OCRv3_mobile_det", rec_model_name


class PPStructureV3CLISubcommandExecutor(PipelineCLISubcommandExecutor):
    @property
    def subparser_name(self):
        return "pp_structurev3"

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

    def execute_with_args(self, args):
        params = get_subcommand_args(args)
        perform_simple_inference(
            PPStructureV3,
            params,
        )
