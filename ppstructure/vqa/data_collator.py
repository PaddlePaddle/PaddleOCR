# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numbers
import numpy as np


class DataCollator:
    """
    data batch
    """

    def __call__(self, batch):
        data_dict = {}
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (np.ndarray, paddle.Tensor, numbers.Number)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = paddle.to_tensor(data_dict[k])
        return data_dict
