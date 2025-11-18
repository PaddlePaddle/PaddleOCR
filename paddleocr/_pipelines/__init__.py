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

from .doc_preprocessor import DocPreprocessor
from .doc_understanding import DocUnderstanding
from .formula_recognition import FormulaRecognitionPipeline
from .ocr import PaddleOCR
from .paddleocr_vl import PaddleOCRVL
from .pp_chatocrv4_doc import PPChatOCRv4Doc
from .pp_doctranslation import PPDocTranslation
from .pp_structurev3 import PPStructureV3
from .seal_recognition import SealRecognition
from .table_recognition_v2 import TableRecognitionPipelineV2

__all__ = [
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
]
