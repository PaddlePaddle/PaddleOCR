# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class TPMRRGN(nn.Layer):
    def __init__(self, in_channels, f_num):
        super(TPMRRGN, self).__init__()
        self.f_num = f_num
        self.conv1 = nn.Sequential(
            nn.Conv2D(
                in_channels,
                in_channels,
                kernel_size=[5, 1],
                stride=1,
                padding=1),
            nn.Conv2D(
                in_channels,
                in_channels,
                kernel_size=[1, 5],
                stride=1,
                padding=1),
            nn.Conv2D(
                in_channels, 1, kernel_size=1, stride=1, padding=0))
        out_channels = in_channels + 1
        self.conv2 = nn.Sequential(
            nn.Conv2D(
                out_channels,
                out_channels,
                kernel_size=[5, 1],
                stride=1,
                padding=1),
            nn.Conv2D(
                out_channels, out_channels, kernel_size=[1, 5], padding=1),
            nn.Conv2D(
                out_channels, 1, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        fea_list = []
        for i in range(self.f_num):
            if i < 1e-3:
                f = self.conv0(x)
                fea_list.append(f)
                continue
            x2 = paddle.concat([x, fea_list[-1]], axis=1)
            f = self.conv2(x2)
            fea_list.append(f)
        out = paddle.concat(fea_list, axis=1)
        return out
