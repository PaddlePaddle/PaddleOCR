# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any


PP_STRUCTUREV3_RUNTIME_PARAMS: dict[str, type] = {
    "use_doc_orientation_classify": bool,
    "use_doc_unwarping": bool,
    "use_textline_orientation": bool,
    "use_seal_recognition": bool,
    "use_table_recognition": bool,
    "use_formula_recognition": bool,
    "use_chart_recognition": bool,
    "use_region_detection": bool,
    "layout_threshold": float,
    "layout_nms": bool,
    "layout_unclip_ratio": float,
    "layout_merge_bboxes_mode": int,
    "text_det_limit_side_len": int,
    "text_det_limit_type": str,
    "text_det_thresh": float,
    "text_det_box_thresh": float,
    "text_det_unclip_ratio": float,
    "text_rec_score_thresh": float,
    "seal_det_limit_side_len": int,
    "seal_det_limit_type": str,
    "seal_det_thresh": float,
    "seal_det_box_thresh": float,
    "seal_det_unclip_ratio": float,
    "seal_rec_score_thresh": float,
    "use_wired_table_cells_trans_to_html": bool,
    "use_wireless_table_cells_trans_to_html": bool,
    "use_table_orientation_classify": bool,
    "use_ocr_results_with_table_cells": bool,
    "use_e2e_wired_table_rec_model": bool,
    "use_e2e_wireless_table_rec_model": bool,
}

PP_STRUCTUREV3_DEFAULT_PARAMS: dict[str, Any] = {
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_chart_recognition": True,
}
