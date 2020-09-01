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

import paddle.fluid as fluid


class ClsLoss(object):
    def __init__(self, params):
        super(ClsLoss, self).__init__()
        self.loss_func = fluid.layers.cross_entropy

    def __call__(self, predicts, labels):
        predict = predicts['predict']
        label = labels['label']
        # softmax_out = fluid.layers.softmax(predict, use_cudnn=False)
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        sum_cost = fluid.layers.mean(cost)
        return sum_cost
