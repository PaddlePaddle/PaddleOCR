# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr

import math
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
ones_ = Constant(value=1.)
zeros_ = Constant(value=0.)


class CT_Head(nn.Layer):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_classes,
                 loss_kernel=None,
                 loss_loc=None):
        super(CT_Head, self).__init__()
        self.conv1 = nn.Conv2D(
            in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(hidden_dim)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2D(
            hidden_dim, num_classes, kernel_size=1, stride=1, padding=0)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                normal_ = Normal(mean=0.0, std=math.sqrt(2. / n))
                normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                zeros_(m.bias)
                ones_(m.weight)

    def _upsample(self, x, scale=1):
        return F.upsample(x, scale_factor=scale, mode='bilinear')

    def forward(self, f, targets=None):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)

        if self.training:
            out = self._upsample(out, scale=4)
            return {'maps': out}
        else:
            score = F.sigmoid(out[:, 0, :, :])
            return {'maps': out, 'score': score}
