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

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


class Model(str, Enum):
    PP_OCRV5 = "PP-OCRv5"
    PP_STRUCTURE_V3 = "PP-StructureV3"
    PADDLE_OCR_VL = "PaddleOCR-VL"
    PADDLE_OCR_VL_15 = "PaddleOCR-VL-1.5"


_OCR_MODELS = frozenset({Model.PP_OCRV5})
_DOCUMENT_PARSING_MODELS = frozenset(
    {Model.PP_STRUCTURE_V3, Model.PADDLE_OCR_VL, Model.PADDLE_OCR_VL_15}
)


def _coerce_model(model: Union[Model, str]) -> Optional[Model]:
    if isinstance(model, Model):
        return model
    try:
        return Model(model)
    except ValueError:
        return None


def is_ocr_model(model: Union[Model, str]) -> bool:
    resolved = _coerce_model(model)
    return resolved in _OCR_MODELS


def is_document_parsing_model(model: Union[Model, str]) -> bool:
    resolved = _coerce_model(model)
    return resolved in _DOCUMENT_PARSING_MODELS


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

    def to_payload(self) -> dict:
        return _build_payload(self)


@dataclass
class DocParsingOptions:
    use_doc_orientation_classify: Optional[bool] = None
    use_doc_unwarping: Optional[bool] = None
    use_textline_orientation: Optional[bool] = None
    use_seal_recognition: Optional[bool] = None
    use_table_recognition: Optional[bool] = None
    use_formula_recognition: Optional[bool] = None
    use_chart_recognition: Optional[bool] = None
    use_region_detection: Optional[bool] = None
    use_layout_detection: Optional[bool] = None
    layout_threshold: Optional[Union[float, dict]] = None
    layout_nms: Optional[bool] = None
    layout_unclip_ratio: Optional[Union[float, list, dict]] = None
    layout_merge_bboxes_mode: Optional[str] = None
    text_det_limit_side_len: Optional[int] = None
    text_det_limit_type: Optional[str] = None
    text_det_thresh: Optional[float] = None
    text_det_box_thresh: Optional[float] = None
    text_det_unclip_ratio: Optional[float] = None
    text_rec_score_thresh: Optional[float] = None
    visualize: Optional[bool] = None

    def to_payload(self) -> dict:
        return _build_payload(self)


_FIELD_NAME_MAP = {
    "use_doc_orientation_classify": "useDocOrientationClassify",
    "use_doc_unwarping": "useDocUnwarping",
    "use_textline_orientation": "useTextlineOrientation",
    "text_det_limit_side_len": "textDetLimitSideLen",
    "text_det_limit_type": "textDetLimitType",
    "text_det_thresh": "textDetThresh",
    "text_det_box_thresh": "textDetBoxThresh",
    "text_det_unclip_ratio": "textDetUnclipRatio",
    "text_rec_score_thresh": "textRecScoreThresh",
    "visualize": "visualize",
    "use_seal_recognition": "useSealRecognition",
    "use_table_recognition": "useTableRecognition",
    "use_formula_recognition": "useFormulaRecognition",
    "use_chart_recognition": "useChartRecognition",
    "use_region_detection": "useRegionDetection",
    "use_layout_detection": "useLayoutDetection",
    "layout_threshold": "layoutThreshold",
    "layout_nms": "layoutNms",
    "layout_unclip_ratio": "layoutUnclipRatio",
    "layout_merge_bboxes_mode": "layoutMergeBboxesMode",
}


def _build_payload(options) -> dict:
    payload = {}
    for field_name, api_name in _FIELD_NAME_MAP.items():
        value = getattr(options, field_name, None)
        if value is not None:
            payload[api_name] = value
    return payload
