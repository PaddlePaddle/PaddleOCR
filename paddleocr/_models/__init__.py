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

from .doc_img_orientation_classification import DocImgOrientationClassification
from .doc_vlm import DocVLM
from .formula_recognition import FormulaRecognition
from .layout_detection import LayoutDetection
from .seal_text_detection import SealTextDetection
from .table_cells_detection import TableCellsDetection
from .table_classification import TableClassification
from .table_structure_recognition import TableStructureRecognition
from .text_detection import TextDetection
from .text_image_unwarping import TextImageUnwarping
from .textline_orientation_classification import TextLineOrientationClassification
from .text_recognition import TextRecognition

__all__ = [
    "DocImgOrientationClassification",
    "DocVLM",
    "FormulaRecognition",
    "LayoutDetection",
    "SealTextDetection",
    "TableCellsDetection",
    "TableClassification",
    "TableStructureRecognition",
    "TextDetection",
    "TextImageUnwarping",
    "TextLineOrientationClassification",
    "TextRecognition",
]
