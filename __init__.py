# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from .paddleocr import (
    PaddleOCR,
    PPStructure,
    draw_ocr,
    draw_structure_result,
    save_structure_res,
    download_with_progressbar,
    sorted_layout_boxes,
    convert_info_docx,
    to_excel,
)
import importlib.metadata as importlib_metadata

try:
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "PaddleOCR",
    "PPStructure",
    "draw_ocr",
    "draw_structure_result",
    "save_structure_res",
    "download_with_progressbar",
    "sorted_layout_boxes",
    "convert_info_docx",
    "to_excel",
]
