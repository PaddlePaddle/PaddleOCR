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


class SRNLoss(object):
    def __init__(self, params):
        super(SRNLoss, self).__init__()
        self.char_num = params['char_num']

    def __call__(self, predicts, others):
        predict = predicts['predict']
        word_predict = predicts['word_out']
        gsrm_predict = predicts['gsrm_out']
        label = others['label']
        lbl_weight = others['lbl_weight']

        casted_label = fluid.layers.cast(x=label, dtype='int64')
        cost_word = fluid.layers.cross_entropy(input=word_predict, label=casted_label)
        cost_gsrm = fluid.layers.cross_entropy(input=gsrm_predict, label=casted_label)
        cost_vsfd = fluid.layers.cross_entropy(input=predict, label=casted_label)

        #cost_word = cost_word * lbl_weight
        #cost_gsrm = cost_gsrm * lbl_weight
        #cost_vsfd = cost_vsfd * lbl_weight

        cost_word = fluid.layers.reshape(x=fluid.layers.reduce_sum(cost_word), shape=[1])
        cost_gsrm = fluid.layers.reshape(x=fluid.layers.reduce_sum(cost_gsrm), shape=[1])
        cost_vsfd = fluid.layers.reshape(x=fluid.layers.reduce_sum(cost_vsfd), shape=[1])

        sum_cost = fluid.layers.sum([cost_word, cost_vsfd * 2.0, cost_gsrm * 0.15])

        #sum_cost = fluid.layers.sum([cost_word * 3.0, cost_vsfd, cost_gsrm * 0.15])
        #sum_cost = cost_word

        #fluid.layers.Print(cost_word,message="word_cost")
        #fluid.layers.Print(cost_vsfd,message="img_cost")
        return [sum_cost,cost_vsfd,cost_word]
        #return [sum_cost, cost_vsfd, cost_word]
