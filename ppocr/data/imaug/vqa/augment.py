# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import os
import sys
import numpy as np
import random
from copy import deepcopy


def order_by_tbyx(ocr_info):
    res = sorted(ocr_info, key=lambda r: (r["bbox"][1], r["bbox"][0]))
    for i in range(len(res) - 1):
        for j in range(i, 0, -1):
            if abs(res[j + 1]["bbox"][1] - res[j]["bbox"][1]) < 20 and \
                    (res[j + 1]["bbox"][0] < res[j]["bbox"][0]):
                tmp = deepcopy(res[j])
                res[j] = deepcopy(res[j + 1])
                res[j + 1] = deepcopy(tmp)
            else:
                break
    return res
