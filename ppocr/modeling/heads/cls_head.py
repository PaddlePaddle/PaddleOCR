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

import math
import paddle
from paddle import nn, ParamAttr
import paddle.nn.functional as F


class ClsHead(nn.Layer):
    """
    Class orientation

    Args:

        params(dict): super parameters for build Class network
    """

    def __init__(self, in_channels, class_dim, **kwargs):
        super(ClsHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2D(1)
        stdv = 1.0 / math.sqrt(in_channels * 1.0)
        self.fc = nn.Linear(
            in_channels,
            class_dim,
            weight_attr=ParamAttr(
                name="fc_0.w_0",
                initializer=nn.initializer.Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name="fc_0.b_0"), )

    def forward(self, x):
        x = self.pool(x)
        x = paddle.reshape(x, shape=[x.shape[0], x.shape[1]])
        x = self.fc(x)
        if not self.training:
            x = F.softmax(x, axis=1)
        return x
