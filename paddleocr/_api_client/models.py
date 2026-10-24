# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional, Union

from ._naming import snake_to_camel
from .errors import InvalidRequestError


class Model(str, Enum):
    PP_OCRV5 = "PP-OCRv5"
    PP_OCRV5_LATIN = "PP-OCRv5-latin"
    PP_OCRV6 = "PP-OCRv6"
    PP_STRUCTURE_V3 = "PP-StructureV3"
    PADDLE_OCR_VL = "PaddleOCR-VL"
    PADDLE_OCR_VL_15 = "PaddleOCR-VL-1.5"
    PADDLE_OCR_VL_16 = "PaddleOCR-VL-1.6"


_OCR_MODELS = frozenset({Model.PP_OCRV5, Model.PP_OCRV5_LATIN, Model.PP_OCRV6})
_DOCUMENT_PARSING_MODELS = frozenset(
    {
        Model.PP_STRUCTURE_V3,
        Model.PADDLE_OCR_VL,
        Model.PADDLE_OCR_VL_15,
        Model.PADDLE_OCR_VL_16,
    }
)
_VL_MODELS = frozenset(
    {
        Model.PADDLE_OCR_VL,
        Model.PADDLE_OCR_VL_15,
        Model.PADDLE_OCR_VL_16,
    }
)


def _coerce_model(model: Union[Model, str]) -> Optional[Model]:
    if isinstance(model, Model):
        return model
    try:
        return Model(model)
    except ValueError:
        return None


def is_ocr_model(model: Union[Model, str]) -> bool:
    return _coerce_model(model) in _OCR_MODELS


def is_document_parsing_model(model: Union[Model, str]) -> bool:
    return _coerce_model(model) in _DOCUMENT_PARSING_MODELS


def is_vl_model(model: Union[Model, str]) -> bool:
    return _coerce_model(model) in _VL_MODELS


@dataclass
class OCROptions:
    use_doc_orientation_classify: Optional[bool] = None
    use_doc_unwarping: Optional[bool] = None
    use_textline_orientation: Optional[bool] = None
    text_det_limit_side_len: Optional[int] = None
    text_det_limit_type: Optional[str] = None
    text_det_thresh: Optional[float] = None
    text_det_box_thresh: Optional[float] = None
    text_det_unclip_ratio: Optional[float] = None
    text_rec_score_thresh: Optional[float] = None
    visualize: Optional[bool] = None
    extra_options: Optional[dict] = None

    def to_payload(self) -> dict:
        return _build_payload(self)


@dataclass
class PPStructureV3Options:
    use_doc_orientation_classify: Optional[bool] = None
    use_doc_unwarping: Optional[bool] = None
    use_textline_orientation: Optional[bool] = None
    use_seal_recognition: Optional[bool] = None
    use_table_recognition: Optional[bool] = None
    use_formula_recognition: Optional[bool] = None
    use_chart_recognition: Optional[bool] = None
    use_region_detection: Optional[bool] = None
    layout_threshold: Optional[Union[float, dict]] = None
    layout_nms: Optional[bool] = None
    layout_unclip_ratio: Optional[Union[float, list, dict]] = None
    layout_merge_bboxes_mode: Optional[Union[str, dict]] = None
    format_block_content: Optional[bool] = None
    text_det_limit_side_len: Optional[int] = None
    text_det_limit_type: Optional[str] = None
    text_det_thresh: Optional[float] = None
    text_det_box_thresh: Optional[float] = None
    text_det_unclip_ratio: Optional[float] = None
    text_rec_score_thresh: Optional[float] = None
    use_wired_table_cells_trans_to_html: Optional[bool] = None
    use_wireless_table_cells_trans_to_html: Optional[bool] = None
    use_table_orientation_classify: Optional[bool] = None
    use_ocr_results_with_table_cells: Optional[bool] = None
    use_e2e_wired_table_rec_model: Optional[bool] = None
    use_e2e_wireless_table_rec_model: Optional[bool] = None
    markdown_ignore_labels: Optional[list] = None
    prettify_markdown: Optional[bool] = None
    show_formula_number: Optional[bool] = None
    return_markdown_images: Optional[bool] = None
    output_formats: Optional[list] = None
    visualize: Optional[bool] = None
    extra_options: Optional[dict] = None

    def to_payload(self) -> dict:
        return _build_payload(self)


@dataclass
class PaddleOCRVLOptions:
    use_doc_orientation_classify: Optional[bool] = None
    use_doc_unwarping: Optional[bool] = None
    use_layout_detection: Optional[bool] = None
    use_chart_recognition: Optional[bool] = None
    use_seal_recognition: Optional[bool] = None
    use_ocr_for_image_block: Optional[bool] = None
    layout_threshold: Optional[Union[float, dict]] = None
    layout_nms: Optional[bool] = None
    layout_unclip_ratio: Optional[Union[float, list, dict]] = None
    layout_merge_bboxes_mode: Optional[Union[str, dict]] = None
    layout_shape_mode: Optional[str] = None
    prompt_label: Optional[str] = None
    format_block_content: Optional[bool] = None
    repetition_penalty: Optional[float] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    max_new_tokens: Optional[int] = None
    vlm_extra_args: Optional[dict] = None
    merge_layout_blocks: Optional[bool] = None
    markdown_ignore_labels: Optional[list] = None
    prettify_markdown: Optional[bool] = None
    show_formula_number: Optional[bool] = None
    restructure_pages: Optional[bool] = None
    merge_tables: Optional[bool] = None
    relevel_titles: Optional[bool] = None
    return_markdown_images: Optional[bool] = None
    output_formats: Optional[list] = None
    visualize: Optional[bool] = None
    extra_options: Optional[dict] = None

    def to_payload(self) -> dict:
        _validate_vl_options(self)
        return _build_payload(self)


DocParsingOptions = Union[PPStructureV3Options, PaddleOCRVLOptions]


def _build_payload(options) -> dict:
    payload = {}
    for field in fields(options):
        value = getattr(options, field.name)
        if value is None:
            continue
        if field.name == "extra_options":
            payload.update(value)
        else:
            payload[snake_to_camel(field.name)] = value
    return payload


def _validate_vl_options(options: PaddleOCRVLOptions) -> None:
    if options.top_p is not None and not (0 < options.top_p <= 1):
        raise InvalidRequestError(
            "top_p must be greater than 0 and less than or equal to 1."
        )
    if options.temperature is not None and options.temperature < 0:
        raise InvalidRequestError("temperature must be greater than or equal to 0.")
    if options.repetition_penalty is not None and options.repetition_penalty <= 0:
        raise InvalidRequestError("repetition_penalty must be greater than 0.")
    if options.min_pixels is not None and options.min_pixels <= 0:
        raise InvalidRequestError("min_pixels must be greater than 0.")
    if options.max_pixels is not None and options.max_pixels <= 0:
        raise InvalidRequestError("max_pixels must be greater than 0.")
    if (
        options.min_pixels is not None
        and options.max_pixels is not None
        and options.min_pixels > options.max_pixels
    ):
        raise InvalidRequestError("min_pixels cannot be greater than max_pixels.")
