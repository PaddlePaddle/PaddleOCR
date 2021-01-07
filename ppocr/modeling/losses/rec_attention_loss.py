#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
import numpy as np


class AttentionLoss(object):
    def __init__(self, params):
        super(AttentionLoss, self).__init__()
        self.char_num = params['char_num']

    def __call__(self, predicts, labels):
        predict = predicts['predict']
        label_out = labels['label_out']
        label_out = fluid.layers.cast(x=label_out, dtype='int64')
        # calculate attention loss
        cost = fluid.layers.cross_entropy(input=predict, label=label_out)
        sum_cost = fluid.layers.reduce_sum(cost)
        return sum_cost
