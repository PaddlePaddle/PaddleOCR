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

from dataclasses import dataclass, field
from enum import Enum


class Model(str, Enum):
    PP_OCRV5 = "PP-OCRv5"
    PP_STRUCTURE_V3 = "PP-StructureV3"
    PADDLE_OCR_VL = "PaddleOCR-VL"
    PADDLE_OCR_VL_15 = "PaddleOCR-VL-1.5"


@dataclass
class OCROptions:
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False

    def to_payload(self) -> dict:
        return {
            "useDocOrientationClassify": self.use_doc_orientation_classify,
            "useDocUnwarping": self.use_doc_unwarping,
            "useTextlineOrientation": self.use_textline_orientation,
        }


@dataclass
class DocParsingOptions:
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_chart_recognition: bool = False

    def to_payload(self) -> dict:
        return {
            "useDocOrientationClassify": self.use_doc_orientation_classify,
            "useDocUnwarping": self.use_doc_unwarping,
            "useChartRecognition": self.use_chart_recognition,
        }
