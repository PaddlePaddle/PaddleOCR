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

PIPELINE_MKLDNN_BLOCKLIST = [
    "formula_recognition",
    "table_recognition_v2",
    "PP-StructureV3",
]

MODEL_MKLDNN_BLOCKLIST = [
    "SLANeXt_wired",
    "SLANeXt_wireless",
    "LaTeX_OCR_rec",
    "PP-FormulaNet-L",
    "PP-FormulaNet-S",
    "UniMERNet",
    "PP-FormulaNet_plus-L",
    "PP-FormulaNet_plus-M",
    "PP-FormulaNet_plus-S",
]
