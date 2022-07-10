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


class DistortBBox:
    def __init__(self, prob=0.5, max_scale=1, **kwargs):
        """Random distort bbox
        """
        self.prob = prob
        self.max_scale = max_scale

    def __call__(self, data):
        if random.random() > self.prob:
            return data
        bbox = np.array(data['bbox'])
        rnd_scale = (np.random.rand(*bbox.shape) - 0.5) * 2 * self.max_scale
        bbox = np.round(bbox + rnd_scale).astype(bbox.dtype)
        data['bbox'] = np.clip(data['bbox'], 0, 1000)
        data['bbox'] = bbox.tolist()
        sys.stdout.flush()
        return data
