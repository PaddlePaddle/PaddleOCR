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
    def __init__(self, epsilon=None):
        super().__init__()
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
        return loss


class KLJSLoss(object):
    def __init__(self, mode='kl'):
        assert mode in ['kl', 'js', 'KL', 'JS'
                        ], "mode can only be one of ['kl', 'js', 'KL', 'JS']"
        self.mode = mode

    def __call__(self, p1, p2, reduction="mean"):

        loss = paddle.multiply(p2, paddle.log((p2 + 1e-5) / (p1 + 1e-5) + 1e-5))

        if self.mode.lower() == "js":
            loss += paddle.multiply(
                p1, paddle.log((p1 + 1e-5) / (p2 + 1e-5) + 1e-5))
            loss *= 0.5
        if reduction == "mean":
            loss = paddle.mean(loss, axis=[1, 2])
        elif reduction == "none" or reduction is None:
            return loss
        else:
            loss = paddle.sum(loss, axis=[1, 2])

        return loss


class DMLLoss(nn.Layer):
    """
    DMLLoss
    """

    def __init__(self, act=None, use_log=False):
        super().__init__()
        if act is not None:
            assert act in ["softmax", "sigmoid"]
        if act == "softmax":
            self.act = nn.Softmax(axis=-1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

        self.use_log = use_log

        self.jskl_loss = KLJSLoss(mode="js")

    def forward(self, out1, out2):
        if self.act is not None:
            out1 = self.act(out1)
            out2 = self.act(out2)
        if self.use_log:
            # for recognition distillation, log is needed for feature map
            log_out1 = paddle.log(out1)
            log_out2 = paddle.log(out2)
            loss = (F.kl_div(
                log_out1, out2, reduction='batchmean') + F.kl_div(
                    log_out2, out1, reduction='batchmean')) / 2.0
        else:
            # for detection distillation log is not needed
            loss = self.jskl_loss(out1, out2)
        return loss


class DistanceLoss(nn.Layer):
    """
    DistanceLoss:
        mode: loss mode
    """

    def __init__(self, mode="l2", **kargs):
        super().__init__()
        assert mode in ["l1", "l2", "smooth_l1"]
        if mode == "l1":
            self.loss_func = nn.L1Loss(**kargs)
        elif mode == "l2":
            self.loss_func = nn.MSELoss(**kargs)
        elif mode == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss(**kargs)

    def forward(self, x, y):
        return self.loss_func(x, y)


class LossFromOutput(nn.Layer):
    def __init__(self, key='loss', reduction='none'):
        super().__init__()
        self.key = key
        self.reduction = reduction

    def forward(self, predicts, batch):
        loss = predicts[self.key]
        if self.reduction == 'mean':
            loss = paddle.mean(loss)
        elif self.reduction == 'sum':
            loss = paddle.sum(loss)
        return {'loss': loss}
