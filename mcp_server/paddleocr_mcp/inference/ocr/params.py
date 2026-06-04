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


OCR_RUNTIME_PARAMS: dict[str, type] = {
    "use_doc_orientation_classify": bool,
    "use_doc_unwarping": bool,
    "use_textline_orientation": bool,
    "text_det_limit_side_len": int,
    "text_det_limit_type": str,
    "text_det_thresh": float,
    "text_det_box_thresh": float,
    "text_det_unclip_ratio": float,
    "text_rec_score_thresh": float,
}

OCR_DEFAULT_PARAMS: dict[str, Any] = {
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
}
