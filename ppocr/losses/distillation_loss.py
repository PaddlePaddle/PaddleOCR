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

import paddle
from paddle import nn
import paddle.nn.functional as F

from .rec_ctc_loss import CTCLoss


class DistillationLoss(nn.Layer):
    def __init__(self,
                 loss_type="celoss",
                 with_basic_loss=False,
                 basic_losc_ratio=0.0,
                 **kwargs):
        super(DistillationLoss, self).__init__()
        self.loss_type = loss_type
        self.with_basic_loss = with_basic_loss
        self.basic_losc_ratio = basic_losc_ratio
        # TODO: add more loss
        supported_loss_type = ["celoss"]
        assert self.loss_type in supported_loss_type, "self.loss_type({}) must be in supported_loss_type({})".format(
            self.loss_type, supported_loss_type)

        if self.with_basic_loss:
            self.basic_loss_func = CTCLoss()

    def __call__(self, predicts, batch):
        teacher_out = predicts["teacher_out"]
        student_out = predicts["student_out"]

        loss_dict = dict()
        if self.loss_type == "celoss":
            y = F.softmax(teacher_out, axis=-1)
            tmp_out = paddle.sum(y, axis=-1)
            cost = F.cross_entropy(student_out, y, soft_label=True)
            cost = paddle.mean(cost)
            loss_dict["celoss"] = cost * (1 - self.basic_losc_ratio)
        else:
            assert False, "not supported loss type!"

        if self.with_basic_loss:
            basic_loss = self.basic_loss_func(
                student_out, batch)["loss"] * self.basic_losc_ratio
            loss_dict["ctcloss"] = basic_loss

        loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict
