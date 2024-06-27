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
from paddle import nn


class SRNLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(SRNLoss, self).__init__()
        self.loss_func = paddle.nn.loss.CrossEntropyLoss(reduction="sum")

    def forward(self, predicts, batch):
        predict = predicts["predict"]
        word_predict = predicts["word_out"]
        gsrm_predict = predicts["gsrm_out"]
        label = batch[1]

        casted_label = paddle.cast(x=label, dtype="int64")
        casted_label = paddle.reshape(x=casted_label, shape=[-1, 1])

        cost_word = self.loss_func(word_predict, label=casted_label)
        cost_gsrm = self.loss_func(gsrm_predict, label=casted_label)
        cost_vsfd = self.loss_func(predict, label=casted_label)

        cost_word = paddle.reshape(x=paddle.sum(cost_word), shape=[1])
        cost_gsrm = paddle.reshape(x=paddle.sum(cost_gsrm), shape=[1])
        cost_vsfd = paddle.reshape(x=paddle.sum(cost_vsfd), shape=[1])

        sum_cost = cost_word * 3.0 + cost_vsfd + cost_gsrm * 0.15

        return {"loss": sum_cost, "word_loss": cost_word, "img_loss": cost_vsfd}
