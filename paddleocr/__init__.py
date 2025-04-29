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

from ._models import (
    DocImgOrientationClassification,
    DocVLM,
    FormulaRecognition,
    LayoutDetection,
    SealTextDetection,
    TableCellsDetection,
    TableClassification,
    TableStructureRecognition,
    TextDetection,
    TextImageUnwarping,
    TextLineOrientationClassification,
    TextRecognition,
)
from ._pipelines import (
    DocPreprocessor,
    DocUnderstanding,
    FormulaRecognitionPipeline,
    PaddleOCR,
    PPChatOCRv4Doc,
    PPStructureV3,
    SealRecognition,
    TableRecognitionPipelineV2,
)
from ._version import version as __version__

__all__ = [
    "DocImgOrientationClassification",
    "DocVLM",
    "FormulaRecognition",
    "SealTextDetection",
    "LayoutDetection",
    "TableCellsDetection",
    "TableClassification",
    "TableStructureRecognition",
    "TextDetection",
    "TextImageUnwarping",
    "TextLineOrientationClassification",
    "TextRecognition",
    "DocPreprocessor",
    "DocUnderstanding",
    "FormulaRecognitionPipeline",
    "PaddleOCR",
    "PPChatOCRv4Doc",
    "PPStructureV3",
    "SealRecognition",
    "TableRecognitionPipelineV2",
    "__version__",
]
