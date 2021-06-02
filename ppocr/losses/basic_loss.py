#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.nn import L1Loss
from paddle.nn import MSELoss as L2Loss
from paddle.nn import SmoothL1Loss


class CELoss(nn.Layer):
    def __init__(self, name="loss_ce", epsilon=None):
        super().__init__()
        self.name = name
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon

    def _labelsmoothing(self, target, class_num):
        if target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def forward(self, x, label):
        loss_dict = {}
        if self.epsilon is not None:
            class_num = x.shape[-1]
            label = self._labelsmoothing(label, class_num)
            x = -F.log_softmax(x, axis=-1)
            loss = paddle.sum(x * label, axis=-1)
        else:
            if label.shape[-1] == x.shape[-1]:
                label = F.softmax(label, axis=-1)
                soft_label = True
            else:
                soft_label = False
            loss = F.cross_entropy(x, label=label, soft_label=soft_label)

        loss_dict[self.name] = paddle.mean(loss)
        return loss_dict


class DMLLoss(nn.Layer):
    """
    DMLLoss
    """

    def __init__(self, name="loss_dml"):
        super().__init__()
        self.name = name

    def forward(self, out1, out2):
        loss_dict = {}
        soft_out1 = F.softmax(out1, axis=-1)
        log_soft_out1 = paddle.log(soft_out1)
        soft_out2 = F.softmax(out2, axis=-1)
        log_soft_out2 = paddle.log(soft_out2)
        loss = (F.kl_div(
            log_soft_out1, soft_out2, reduction='batchmean') + F.kl_div(
                log_soft_out2, soft_out1, reduction='batchmean')) / 2.0
        loss_dict[self.name] = loss
        return loss_dict


class DistanceLoss(nn.Layer):
    """
    DistanceLoss:
        mode: loss mode
        name: loss key in the output dict
    """

    def __init__(self, mode="l2", name="loss_dist", **kargs):
        assert mode in ["l1", "l2", "smooth_l1"]
        if mode == "l1":
            self.loss_func = nn.L1Loss(**kargs)
        elif mode == "l1":
            self.loss_func = nn.MSELoss(**kargs)
        elif mode == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss(**kargs)

        self.name = "{}_{}".format(name, mode)

    def forward(self, x, y):
        return {self.name: self.loss_func(x, y)}
