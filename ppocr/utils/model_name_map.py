# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

MODELS_DICT = {
    "PP-OCRv4_server_rec": ["ch_PP-OCRv4_rec_hgnet"],
    "PP-OCRv4_mobile_rec": ["ch_PP-OCRv4_rec"],
    "PP-OCRv4_mobile_rec": ["ch_PP-OCRv4_rec"],
    "PP-OCRv3_mobile_rec": [
        "ch_PP-OCRv3_rec",
    ],
    "en_PP-OCRv3_mobile_rec": [
        "en_PP-OCRv3_rec",
    ],
    "PP-OCRv4_server_rec_doc": ["ch_PP-OCRv4_rec_hgnet_doc"],
    "PP-OCRv4_mobile_det": ["ch_PP-OCRv4_det_student"],
    "PP-OCRv4_server_det": ["ch_PP-OCRv4_det_teacher"],
}
