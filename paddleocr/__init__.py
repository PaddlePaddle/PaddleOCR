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

from paddlex.inference.utils.benchmark import benchmark

from ._models import (
    ChartParsing,
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
    PaddleOCRVL,
    PPChatOCRv4Doc,
    PPDocTranslation,
    PPStructureV3,
    SealRecognition,
    TableRecognitionPipelineV2,
)
from ._utils.logging import logger
from ._version import version as __version__


def doc2md_convert(source, **kwargs):
    """Convert an office document to Markdown. See paddleocr._doc2md.convert."""
    from ._doc2md import convert

    return convert(source, **kwargs)


def doc2md_supported_formats():
    """Return supported file extensions. See paddleocr._doc2md.supported_formats."""
    from ._doc2md import supported_formats

    return supported_formats()


__all__ = [
    "benchmark",
    "ChartParsing",
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
    "PaddleOCRVL",
    "PPChatOCRv4Doc",
    "PPDocTranslation",
    "PPStructureV3",
    "SealRecognition",
    "TableRecognitionPipelineV2",
    "doc2md_convert",
    "doc2md_supported_formats",
    "logger",
    "__version__",
]
