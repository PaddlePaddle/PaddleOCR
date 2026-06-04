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


PADDLEOCR_VL_RUNTIME_PARAMS: dict[str, type] = {
    "use_doc_orientation_classify": bool,
    "use_doc_unwarping": bool,
    "use_layout_detection": bool,
    "use_chart_recognition": bool,
    "use_seal_recognition": bool,
    "layout_threshold": float,
    "layout_nms": bool,
    "layout_unclip_ratio": float,
    "layout_merge_bboxes_mode": int,
    "layout_shape_mode": str,
    "prompt_label": str,
    "repetition_penalty": float,
    "temperature": float,
    "top_p": float,
    "min_pixels": int,
    "max_pixels": int,
    "max_new_tokens": int,
}

PADDLEOCR_VL_DEFAULT_PARAMS: dict[str, Any] = {
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_chart_recognition": True,
    "use_seal_recognition": True,
}
